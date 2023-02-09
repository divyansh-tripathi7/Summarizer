library(textrank)
data(joboffer)
cat(unique(joboffer$sentence), sep = "\n")
head(joboffer[, c("sentence_id", "lemma", "upos")], 10)

job_rawtxt <- readLines(system.file(package = "textrank", "extdata", "joboffer.txt"))
job_rawtxt <- paste(job_rawtxt, collapse = "\n")

library(udpipe)
tagger <- udpipe_download_model("english")
tagger <- udpipe_load_model(tagger$file_model)
joboffer <- udpipe_annotate(tagger, job_rawtxt)
joboffer <- as.data.frame(joboffer)

keyw <- textrank_keywords(joboffer$lemma,
                          relevant = joboffer$upos %in% c("NOUN", "VERB", "ADJ"))
subset(keyw$keywords, ngram > 1 & freq > 1)

head(joboffer[, c("sentence_id", "lemma", "upos")], 10)

library(udpipe)
joboffer$textrank_id <- unique_identifier(joboffer, c("doc_id", "paragraph_id", "sentence_id"))
sentences <- unique(joboffer[, c("textrank_id", "sentence")])
terminology <- subset(joboffer, upos %in% c("NOUN", "ADJ"))
terminology <- terminology[, c("textrank_id", "lemma")]
head(terminology)

## Textrank for finding the most relevant sentences
tr <- textrank_sentences(data = sentences, terminology = terminology)
names(tr)

plot(sort(tr$pagerank$vector, decreasing = TRUE), 
     type = "b", ylab = "Pagerank", main = "Textrank")

