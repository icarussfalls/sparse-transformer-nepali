from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# Optional: clean punctuation to match better
def normalize(text):
    return text.replace("“", "").replace("”", "").replace(",", "").replace(".", "").replace("।", "").strip()

pred = "तर म ती मानिसहरूलाई । म ती मानिसहरूलाई ।” परमप्रभु भन्नुहुन्छ ।"
ref =  "त्यसपछि अम्मोनीहरूबाट जे खोसिएको थियो ती सबै कैदीहरूलाई फर्काएर ल्याउनेछु।” यो सन्देश परमप्रभुबाट आयो।"


# Normalize and tokenize
pred_tok = normalize(pred).split()
ref_tok = normalize(ref).split()

# BLEU with smoothing (needed for short or bad predictions)
smoothie = SmoothingFunction().method4
score = sentence_bleu([ref_tok], pred_tok, smoothing_function=smoothie)

print(f"BLEU Score: {score:.4f}")



def normalize1(text):
    return text.replace("“", "").replace("”", "").replace(",", "").replace(".", "").replace("।", "").strip()
pred_norm = [normalize1(p) for p in predicted]
ref_norm = [normalize1(r) for r in expected]

smoothie = SmoothingFunction().method4
# BLEU
bleu = sentence_bleu([ref_norm], pred_norm, smoothing_function=smoothie)