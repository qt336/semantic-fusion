# Gemma 4 31B IT 分词核对

使用 Google 发布的 [tokenizer.json](https://huggingface.co/google/gemma-4-31B-it/blob/main/tokenizer.json)，在完整原始 user 消息中核对开头序列的字符位置。英文把前导分隔空格计入后一个词元。逗号条件不要求三 token。此核对未包含中转平台的聊天模板，也不适用于 Gemini 3.6 Flash。

分词器 SHA-256 为 `cc8d3a0ce36466ccc1278bf987df5f71db1719b9ca6b4118264f45cb627bfe0f`。

|语言|条件|三个 token 且边界对齐|总数|
|---|---|---:|---:|
|zh|semantic|500|500|
|zh|random|500|500|
|en|semantic|500|500|
|en|random|500|500|
