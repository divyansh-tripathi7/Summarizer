from sumy.summarizers.lsa
import LsaSummarizer
def lsa_method(text):
  parser = PlaintextParser.from_string(text, Tokenizer("english"))
summarizer_lsa = LsaSummarizer()
summary_2 = summarizer_lsa(parser.document, 2)
dp = []
for i in summary_2:
  lp = str(i)
dp.append(lp)
final_sentence = ' '.join(dp)
return final_sentence
