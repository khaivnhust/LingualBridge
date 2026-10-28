// mt_prompt_dump.cc -- write the exact MT prompt contract the harness sends, as JSON,
// straight from pipeline.h (so a handoff can never drift from the code):
//   { "<tgt>": { "user_content_template": "...{source}...", "anchor": "...",
//                "example_user_content": "..." }, ... }
// Host build: g++ -std=c++17 -Isrc src/mt_prompt_dump.cc src/http.cc -o mt_prompt_dump
#include <cstdio>
#include <string>

#include "http.h"
#include "pipeline.h"

using namespace s2st;

int main() {
    const std::string PH = "{source}";
    std::printf("{\n");
    const char * tgts[] = {"en", "zh", "ko", "vi"};
    for (int i = 0; i < 4; ++i) {
        const std::string t = tgts[i];
        std::printf("  \"%s\": {\n", t.c_str());
        std::printf("    \"user_content_template\": \"%s\",\n", json_escape(mt_prompt(PH, t, {})).c_str());
        std::printf("    \"anchor\": \"%s\",\n", json_escape(mt_anchors().at(t)).c_str());
        std::printf("    \"example_user_content\": \"%s\"\n",
                    json_escape(mt_prompt("We need European leadership and solutions right now.", t, {})).c_str());
        std::printf("  }%s\n", i < 3 ? "," : "");
    }
    std::printf("}\n");
    return 0;
}
