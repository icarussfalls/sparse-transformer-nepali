import matplotlib.pyplot as plt
import pandas as pd

# Data from logs
epochs = list(range(1, 21))

vanilla_bleu = [0.0000, 0.0000, 0.0001, 0.0013, 0.0000, 0.0016, 0.0072, 0.0209, 0.0004, 0.0136,
                0.0028, 0.0062, 0.0077, 0.0126, 0.0126, 0.0256, 0.0101, 0.0613, 0.0110, 0.0217]
vanilla_cer = [1.0000, 0.9766, 0.9380, 0.8792, 0.8049, 0.8974, 0.7546, 0.7546, 0.8384, 0.5957,
               0.8119, 0.6829, 0.7666, 0.8034, 0.7596, 0.5983, 0.8016, 0.7044, 0.8251, 0.7393]
vanilla_wer = [1.0000, 1.0000, 0.9770, 0.9730, 1.0000, 0.9211, 0.9762, 0.8889, 0.9808, 0.8571,
               0.9333, 0.9344, 0.9348, 1.1481, 1.0606, 0.8621, 1.2424, 0.9655, 1.2667, 0.9375]

modified_bleu = [0.0000, 0.0000, 0.0000, 0.0005, 0.0000, 0.0146, 0.0067, 0.0069, 0.0096, 0.0000,
                 0.0092, 0.0024, 0.0030, 0.0000, 0.0432, 0.0098, 0.0102, 0.0187, 0.0022, 0.0246]
modified_cer = [1.0000, 0.9594, 0.9754, 0.9202, 0.9799, 0.8834, 0.8085, 0.7548, 0.7727, 0.7976,
                0.7608, 0.8146, 0.8608, 0.8054, 0.6599, 0.6882, 0.6913, 0.7354, 0.7638, 0.6803]
modified_wer = [1.0000, 1.0000, 1.0000, 0.9744, 1.0000, 1.0000, 0.9667, 0.9412, 1.0000, 1.0000,
                0.9167, 0.9608, 0.9697, 1.0000, 0.8000, 0.8864, 0.8889, 0.9592, 0.9286, 0.9714]

# Create a stable, clear plot
fig, axs = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

axs[0].plot(epochs, vanilla_bleu, label='Vanilla', marker='o')
axs[0].plot(epochs, modified_bleu, label='Modified', marker='x')
axs[0].set_ylabel("BLEU Score")
axs[0].legend()
axs[0].grid(True)

axs[1].plot(epochs, vanilla_cer, label='Vanilla', marker='o')
axs[1].plot(epochs, modified_cer, label='Modified', marker='x')
axs[1].set_ylabel("CER")
axs[1].legend()
axs[1].grid(True)

axs[2].plot(epochs, vanilla_wer, label='Vanilla', marker='o')
axs[2].plot(epochs, modified_wer, label='Modified', marker='x')
axs[2].set_xlabel("Epochs")
axs[2].set_ylabel("WER")
axs[2].legend()
axs[2].grid(True)

plt.suptitle("Performance Comparison: Vanilla vs Modified Model", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()