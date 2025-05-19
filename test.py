import re
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

def normalize(text):
    # Remove weird quotes, multiple commas, punctuations, extra spaces
    text = text.replace("“", "").replace("”", "")
    text = re.sub(r'[,.।]+', ' ', text)  # replace punctuations with space
    text = re.sub(r'\s+', ' ', text)     # collapse multiple spaces
    return text.strip()

pred = "“ परमप्रभु , परमप्रभु , परमप्रभु , परमप्रभु , “ परमप्रभु , , , , , , , , , , , ,"
ref = "परमप्रभु, हाम्रा पुर्खाका परमेश्वरको स्तुति गर। उसले राजा र यरूशलेमका परमप्रभुको मन्दिरलाई यस प्रकारले सम्मानित गर्ने तुल्याए।"

pred_norm = normalize(pred)
ref_norm = normalize(ref)

pred_tok = pred_norm.split()
ref_tok = ref_norm.split()

smoothie = SmoothingFunction().method4
score = sentence_bleu([ref_tok], pred_tok, smoothing_function=smoothie)

print(f"Normalized Prediction Tokens: {pred_tok}")
print(f"Reference Tokens: {ref_tok}")
print(f"BLEU Score: {score:.4f}")
