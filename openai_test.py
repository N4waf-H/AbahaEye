import openai

# Replace with your actual API key
openai.api_key = "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"

try:
    models = openai.models.list()
    print("API is active! Models available:")
    for m in models.data:
        print("-", m.id)
except Exception as e:
    print("API not active or billing not set up.")
    print("Error:", e)
