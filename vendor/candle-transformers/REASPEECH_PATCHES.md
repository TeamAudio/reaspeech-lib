# ReaSpeech Candle patches

This is `candle-transformers` 0.11.0 from crates.io, vendored so ReaSpeech can
carry some Whisper changes that are not available in the released crate:

- incremental decoder self-attention KV caching;
- decoder KV-cache batch reordering for batched beam search;
- cross-attention capture and dynamic-time-warp word alignment.

The implementation is forward-ported from Hugging Face Candle pull request
[#2728](https://github.com/huggingface/candle/pull/2728). ReaSpeech enables the
KV cache for all transcription and enables attention capture only when word
timestamps are requested.

Local differences from the pull request include compatibility with Candle
0.11 and chronological concatenation of cached keys and values.
