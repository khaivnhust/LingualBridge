#include "http.h"

#include <arpa/inet.h>
#include <cctype>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>

#include <chrono>
#include <sstream>

namespace s2st {
namespace {

using clk = std::chrono::steady_clock;

long long ms_since(const clk::time_point & t0) {
    return std::chrono::duration_cast<std::chrono::milliseconds>(clk::now() - t0).count();
}

int connect_to(const HttpOptions & opt, long long * t_conn_ms, const clk::time_point & t0) {
    int fd = ::socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) return -1;

    int one = 1;
    ::setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));

    sockaddr_in a{};
    a.sin_family = AF_INET;
    a.sin_port   = htons((uint16_t)opt.port);
    if (::inet_pton(AF_INET, opt.host.c_str(), &a.sin_addr) != 1) {
        ::close(fd);
        return -1;
    }

    // non-blocking connect so we can honour connect_timeout_ms
    int flags = ::fcntl(fd, F_GETFL, 0);
    ::fcntl(fd, F_SETFL, flags | O_NONBLOCK);

    int rc = ::connect(fd, (sockaddr *)&a, sizeof(a));
    if (rc != 0 && errno != EINPROGRESS) {
        ::close(fd);
        return -1;
    }
    if (rc != 0) {
        pollfd p{fd, POLLOUT, 0};
        int pr = ::poll(&p, 1, opt.connect_timeout_ms);
        if (pr <= 0) {
            ::close(fd);
            return -1;
        }
        int err = 0;
        socklen_t len = sizeof(err);
        ::getsockopt(fd, SOL_SOCKET, SO_ERROR, &err, &len);
        if (err != 0) {
            ::close(fd);
            return -1;
        }
    }
    ::fcntl(fd, F_SETFL, flags);  // back to blocking; we poll() before each read
    if (t_conn_ms) *t_conn_ms = ms_since(t0);
    return fd;
}

bool write_all(int fd, const char * p, size_t n) {
    size_t off = 0;
    while (off < n) {
        ssize_t w = ::send(fd, p + off, n - off, MSG_NOSIGNAL);
        if (w <= 0) {
            if (w < 0 && (errno == EINTR || errno == EAGAIN)) continue;
            return false;
        }
        off += (size_t)w;
    }
    return true;
}

std::string build_request(const HttpOptions & opt, const std::string & path,
                          const std::string & headers, const std::string & body,
                          const char * method = "POST") {
    std::ostringstream o;
    o << method << " " << path << " HTTP/1.1\r\n"
      << "Host: " << opt.host << ":" << opt.port << "\r\n"
      << headers
      << "Content-Length: " << body.size() << "\r\n"
      << "Connection: close\r\n"
      << "\r\n";
    std::string head = o.str();
    return head + body;
}

// Read until EOF, honouring timeout_ms. Calls on_chunk (if set) for each block.
bool read_all(int fd, const HttpOptions & opt, std::string * out,
              long long * t_first_byte_ms, const clk::time_point & t0,
              const std::function<bool(const char *, size_t)> & on_chunk) {
    char    buf[16384];
    bool    got_first = false;
    long long deadline = ms_since(t0) + opt.timeout_ms;
    for (;;) {
        long long left = deadline - ms_since(t0);
        if (left <= 0) return false;
        if (opt.cancel && opt.cancel->load()) return false;
        // short poll when cancellable, so an abort is noticed within ~5 ms
        const int slice = opt.cancel ? 5 : 1000;
        pollfd p{fd, POLLIN, 0};
        int pr = ::poll(&p, 1, (int)(left > slice ? slice : left));
        if (pr < 0) {
            if (errno == EINTR) continue;
            return false;
        }
        if (pr == 0) continue;
        ssize_t n = ::recv(fd, buf, sizeof(buf), 0);
        if (n < 0) {
            if (errno == EINTR || errno == EAGAIN) continue;
            return false;
        }
        if (n == 0) break;  // EOF
        if (!got_first) {
            got_first = true;
            if (t_first_byte_ms) *t_first_byte_ms = ms_since(t0);
        }
        if (on_chunk) {
            if (!on_chunk(buf, (size_t)n)) return true;  // caller aborted
        } else if (out) {
            out->append(buf, (size_t)n);
        }
    }
    return true;
}

// Split "HTTP/1.1 200 OK\r\n...\r\n\r\n" off the front.
void strip_headers(std::string & raw) {
    size_t p = raw.find("\r\n\r\n");
    if (p == std::string::npos) return;
    raw.erase(0, p + 4);
}

int parse_status(const std::string & raw) {
    size_t sp = raw.find(' ');
    if (sp == std::string::npos) return 0;
    return std::atoi(raw.c_str() + sp + 1);
}

} // namespace

HttpResponse http_post_json(const std::string & path, const std::string & json_body,
                            const HttpOptions & opt) {
    HttpResponse r;
    auto t0 = clk::now();
    int fd = connect_to(opt, nullptr, t0);
    if (fd < 0) { r.status = -1; return r; }

    std::string req = build_request(opt, path, "Content-Type: application/json\r\n", json_body);
    if (!write_all(fd, req.data(), req.size())) { ::close(fd); r.status = -1; return r; }

    std::string raw;
    read_all(fd, opt, &raw, &r.t_first_byte_ms, t0, nullptr);
    ::close(fd);
    r.status   = parse_status(raw);
    strip_headers(raw);
    r.body     = raw;
    return r;
}

HttpResponse http_get(const std::string & path, const HttpOptions & opt) {
    HttpResponse r;
    auto t0 = clk::now();
    int fd = connect_to(opt, nullptr, t0);
    if (fd < 0) { r.status = -1; return r; }
    std::string req = build_request(opt, path, "", "", "GET");
    if (!write_all(fd, req.data(), req.size())) { ::close(fd); r.status = -1; return r; }
    std::string raw;
    read_all(fd, opt, &raw, &r.t_first_byte_ms, t0, nullptr);
    ::close(fd);
    r.status = parse_status(raw);
    strip_headers(raw);
    r.body   = raw;
    return r;
}

HttpResponse http_post_multipart(const std::string & path,
                                 const std::vector<std::pair<std::string, std::string>> & fields,
                                 const std::string & file_field,
                                 const std::string & file_name,
                                 const std::string & mime,
                                 const std::string & file_bytes,
                                 const HttpOptions & opt) {
    // Boundary that cannot appear in JSON or WAV data.
    const std::string B = "----s2stbench7f3a1c9e2b";
    std::string body;
    body.reserve(file_bytes.size() + 4096);
    for (const auto & kv : fields) {
        body += "--" + B + "\r\n";
        body += "Content-Disposition: form-data; name=\"" + kv.first + "\"\r\n\r\n";
        body += kv.second + "\r\n";
    }
    body += "--" + B + "\r\n";
    body += "Content-Disposition: form-data; name=\"" + file_field +
            "\"; filename=\"" + file_name + "\"\r\n";
    body += "Content-Type: " + mime + "\r\n\r\n";
    body += file_bytes;
    body += "\r\n--" + B + "--\r\n";

    std::string headers = "Content-Type: multipart/form-data; boundary=" + B + "\r\n";

    HttpResponse r;
    auto t0 = clk::now();
    int fd = connect_to(opt, nullptr, t0);
    if (fd < 0) { r.status = -1; return r; }
    std::string req = build_request(opt, path, headers, body);
    if (!write_all(fd, req.data(), req.size())) { ::close(fd); r.status = -1; return r; }
    std::string raw;
    read_all(fd, opt, &raw, &r.t_first_byte_ms, t0, nullptr);
    ::close(fd);
    r.status = parse_status(raw);
    strip_headers(raw);
    r.body   = raw;
    return r;
}

StreamStats http_post_stream(const std::string & path, const std::string & json_body,
                             const HttpOptions & opt,
                             const std::function<bool(const std::string &)> & on_delta) {
    StreamStats st;
    auto t0 = clk::now();
    int fd = connect_to(opt, &st.t_connect_ms, t0);
    if (fd < 0) return st;

    std::string req = build_request(opt, path,
                                    "Content-Type: application/json\r\n"
                                    "Accept: text/event-stream\r\n", json_body);
    if (!write_all(fd, req.data(), req.size())) { ::close(fd); return st; }

    // Incremental SSE parsing: accumulate, split on blank line, take `data:`.
    std::string buf;
    auto handle = [&](const char * p, size_t n) -> bool {
        buf.append(p, n);
        st.bytes += n;
        for (;;) {
            size_t eol = buf.find("\n\n");
            if (eol == std::string::npos) break;
            std::string ev = buf.substr(0, eol);
            buf.erase(0, eol + 2);
            size_t d = ev.find("data:");
            if (d == std::string::npos) continue;
            std::string payload = ev.substr(d + 5);
            while (!payload.empty() && (payload.front() == ' ' || payload.front() == '\r'))
                payload.erase(payload.begin());
            if (payload.empty()) continue;
            st.n_chunks++;
            if (st.t_first_chunk_ms < 0) st.t_first_chunk_ms = ms_since(t0);
            st.last_event = payload;
            if (payload != "[DONE]" && !on_delta(payload)) return false;  // early abort
        }
        return true;
    };

    read_all(fd, opt, nullptr, nullptr, t0, handle);
    ::close(fd);
    st.t_done_ms = ms_since(t0);
    return st;
}

std::string json_escape(const std::string & s) {
    std::string o;
    o.reserve(s.size() + 16);
    for (unsigned char c : s) {
        switch (c) {
            case '"':  o += "\\\""; break;
            case '\\': o += "\\\\"; break;
            case '\n': o += "\\n";  break;
            case '\r': o += "\\r";  break;
            case '\t': o += "\\t";  break;
            default:
                if (c < 0x20) { char b[8]; std::snprintf(b, sizeof b, "\\u%04x", c); o += b; }
                else o += (char)c;
        }
    }
    return o;
}

// Encode a Unicode code point as UTF-8. Required because llama-server may emit
// \uXXXX escapes for non-ASCII; three of our four languages are non-Latin, so
// dropping these would silently corrupt every Vietnamese/Korean/Chinese output.
static void utf8_encode(uint32_t cp, std::string * out) {
    if (cp < 0x80) {
        *out += (char)cp;
    } else if (cp < 0x800) {
        *out += (char)(0xC0 | (cp >> 6));
        *out += (char)(0x80 | (cp & 0x3F));
    } else if (cp < 0x10000) {
        *out += (char)(0xE0 | (cp >> 12));
        *out += (char)(0x80 | ((cp >> 6) & 0x3F));
        *out += (char)(0x80 | (cp & 0x3F));
    } else {
        *out += (char)(0xF0 | (cp >> 18));
        *out += (char)(0x80 | ((cp >> 12) & 0x3F));
        *out += (char)(0x80 | ((cp >> 6) & 0x3F));
        *out += (char)(0x80 | (cp & 0x3F));
    }
}

static bool hex4(const std::string & s, size_t i, uint32_t * v) {
    if (i + 4 > s.size()) return false;
    uint32_t x = 0;
    for (int k = 0; k < 4; ++k) {
        char c = s[i + k];
        x <<= 4;
        if (c >= '0' && c <= '9') x |= (uint32_t)(c - '0');
        else if (c >= 'a' && c <= 'f') x |= (uint32_t)(c - 'a' + 10);
        else if (c >= 'A' && c <= 'F') x |= (uint32_t)(c - 'A' + 10);
        else return false;
    }
    *v = x;
    return true;
}

// Minimal scanner for "key" : "value" or "key" : true/false at any depth.
// Good enough for llama-server responses, which are flat inside choices[].
std::string json_get_string(const std::string & j, const std::string & key) {
    std::string pat = "\"" + key + "\"";
    size_t k = j.find(pat);
    if (k == std::string::npos) return "";
    size_t c = j.find(':', k + pat.size());
    if (c == std::string::npos) return "";
    size_t q = j.find('"', c + 1);
    if (q == std::string::npos) return "";
    std::string out;
    for (size_t i = q + 1; i < j.size(); ++i) {
        char ch = j[i];
        if (ch == '\\' && i + 1 < j.size()) {
            char n = j[++i];
            switch (n) {
                case 'n': out += '\n'; break;
                case 't': out += '\t'; break;
                case 'r': out += '\r'; break;
                case 'b': out += '\b'; break;
                case 'f': out += '\f'; break;
                case '/': out += '/';  break;
                case 'u': {
                    uint32_t cp = 0;
                    if (!hex4(j, i + 1, &cp)) { out += '?'; break; }
                    i += 4;
                    // surrogate pair for anything outside the BMP
                    if (cp >= 0xD800 && cp <= 0xDBFF && i + 6 < j.size() &&
                        j[i + 1] == '\\' && j[i + 2] == 'u') {
                        uint32_t lo = 0;
                        if (hex4(j, i + 3, &lo) && lo >= 0xDC00 && lo <= 0xDFFF) {
                            cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                            i += 6;
                        }
                    }
                    utf8_encode(cp, &out);
                    break;
                }
                default: out += n;
            }
            continue;
        }
        if (ch == '"') break;
        out += ch;
    }
    return out;
}

bool json_get_bool(const std::string & j, const std::string & key) {
    std::string pat = "\"" + key + "\"";
    size_t k = j.find(pat);
    if (k == std::string::npos) return false;
    size_t c = j.find(':', k + pat.size());
    if (c == std::string::npos) return false;
    size_t i = c + 1;
    while (i < j.size() && std::isspace((unsigned char)j[i])) ++i;
    return i < j.size() && j.compare(i, 4, "true") == 0;
}

} // namespace s2st
