# Preprocessing Vietnamese crawled data

## Steps:

1. Merge all topic csv together

2. Cleaning steps include:

   - [ ] Lower casing
   - [ ] Removal of Stopwords and particular words
   - [ ] Spelling correction and acronyms, abbreviations lookup
   - [ ] Text normalization ("luônnn" -> "luôn" (always) )
   - [ ] Conversion of emojis to vietnamese words
   - [ ] Conversion of emoticons to vietnamese words
   - [ ] Removal of URLs
   - [ ] Removal of HTML tags
   - [ ] Spelling correction

3. Merge topcis in to 10 types:

   - news
   - sports
   - entertainment,
   - gaming
   - science
   - technology
   - finance,
   - healthcare
   - life
   - education

4. Parts of Speech Tagging


## References
- https://medium.com/geekculture/text-preprocessing-how-to-handle-emoji-emoticon-641bbfa6e9e7
- https://www.kaggle.com/datasets/heeraldedhia/stop-words-in-28-languages?resource=download
