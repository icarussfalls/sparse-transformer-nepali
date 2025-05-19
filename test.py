from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# Optional: clean punctuation to match better
def normalize(text):
    return text.replace("“", "").replace("”", "").replace(",", "").replace(".", "").replace("।", "").strip()

pred = "“ परमप्रभु , परमप्रभु , परमप्रभु , परमप्रभु , “ परमप्रभु , , , , , , , , , , , ,"
ref = "परमप्रभु, हाम्रा पुर्खाका परमेश्वरको स्तुति गर। उसले राजा र यरूशलेमका परमप्रभुको मन्दिरलाई यस प्रकारले सम्मानित गर्ने तुल्याए।"

# Normalize and tokenize
pred_tok = normalize(pred).split()
ref_tok = normalize(ref).split()

# BLEU with smoothing (needed for short or bad predictions)
smoothie = SmoothingFunction().method4
score = sentence_bleu([ref_tok], pred_tok, smoothing_function=smoothie)

print(f"BLEU Score: {score:.4f}")