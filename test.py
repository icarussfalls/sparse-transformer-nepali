from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# Optional: clean punctuation to match better
def normalize(text):
    return text.replace("“", "").replace("”", "").replace(",", "").replace(".", "").replace("।", "").strip()

pred = "तर यदि कुनै मानिसले यो कुरा गरेको कुरा गरेको थियो भने उसले त्यो मानिस । तर उसले त्यो मानिस । तर उसले त्यो मानिस ।"
ref =  "यो कुरा सम्झी राखः घरका मालिकले कति बेला चोर आउँछ भन्ने कुरा चाल पायो भने उसले चोरलाई घर फोर्नु दिएको हुँदैन।"


# Normalize and tokenize
pred_tok = normalize(pred).split()
ref_tok = normalize(ref).split()

# BLEU with smoothing (needed for short or bad predictions)
smoothie = SmoothingFunction().method4
score = sentence_bleu([ref_tok], pred_tok, smoothing_function=smoothie)

print(f"BLEU Score: {score:.4f}")