// http.h -- minimal POSIX HTTP/1.1 client with streaming reads.
//
// Deliberately dependency-free (no libcurl, no nlohmann): the QDC rootfs is a
// stripped Debian container and we already know socket() works there. Two call
// patterns are needed:
//
//   http_post_json()  -- one-shot, Content-Length body (ASR upload, MT non-stream)
//   http_post_stream()-- POST with stream:true and a per-SSE-chunk callback (MT)
#pragma once
#include <atomic>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace s2st {

struct HttpResponse {
    int         status   = 0;
    std::string body;
    long long   t_first_byte_ms = -1;  // connect+send done -> first body byte
};

struct HttpOptions {
    int         port            = 80;
    std::string host            = "127.0.0.1";
    int         timeout_ms      = 600000;
    int         connect_timeout_ms = 5000;
    // When set and raised, the request is abandoned and the socket closed.
    // llama-server cancels a task whose client has gone, but only between
    // decode steps -- audio encoding/prefill already under way still runs out.
    const std::atomic<bool> * cancel = nullptr;
};

// POST application/json. Returns the full body.
HttpResponse http_post_json(const std::string & path,
                            const std::string & json_body,
                            const HttpOptions & opt);

// GET, full body (used for /props).
HttpResponse http_get(const std::string & path, const HttpOptions & opt);

// POST multipart/form-data (used for the ASR audio upload). `fields` are plain
// text fields; `file_field`/`file_name`/`file_bytes` are the audio part.
HttpResponse http_post_multipart(const std::string & path,
                                 const std::vector<std::pair<std::string, std::string>> & fields,
                                 const std::string & file_field,
                                 const std::string & file_name,
                                 const std::string & mime,
                                 const std::string & file_bytes,
                                 const HttpOptions & opt);

// POST with a streaming callback. `on_delta` is called with each SSE `data:`
// payload as it arrives; return false to abort early. Used for MT streaming so
// the caller can hand the first clause to TTS while the rest is still decoding.
struct StreamStats {
    long long t_connect_ms   = -1;   // socket connected + request written
    long long t_first_chunk_ms = -1; // first SSE data: line received
    long long t_done_ms      = -1;   // stream closed by server
    size_t    n_chunks       = 0;
    size_t    bytes          = 0;
    std::string last_event;         // final data: payload (usually [DONE])
};
StreamStats http_post_stream(const std::string & path,
                             const std::string & json_body,
                             const HttpOptions & opt,
                             const std::function<bool(const std::string &)> & on_delta);

// ── tiny JSON string field extractor ────────────────────────────────────────
// Avoids a JSON dependency. Handles escaped characters in the value.
std::string json_get_string(const std::string & json, const std::string & key);
// Handles "stream":true / "stream": false
bool json_get_bool(const std::string & json, const std::string & key);

std::string json_escape(const std::string & s);

} // namespace s2st
