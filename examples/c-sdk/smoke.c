#include <reaspeech.h>
#include <string.h>

int main(void) {
    char message[256] = {0};
    /* Invalid options must fail without downloading models or starting inference. */
    if (reaspeech_start_ex("missing.wav", "{}", message, sizeof message)) return 1;
    if (!message[0]) return 2;
    if (reaspeech_start(NULL, NULL, NULL, false, false, false, NULL, message, sizeof message)) return 3;
    if (reaspeech_start_from_memory(NULL, 0, 16000, 1, "{}", message, sizeof message)) return 4;
    if (strcmp(reaspeech_poll("unknown-job"), "") != 0) return 5;
    if (reaspeech_cancel("unknown-job") != 0) return 6;
    return 0;
}
