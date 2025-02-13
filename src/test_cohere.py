import cohere

co = cohere.ClientV2("qk7nYJiskTU4GHp8pagEYMxkSVIDT7uOraEskRN1")

response = co.chat(
    model="c4ai-aya-expanse-32b",
    messages=[
        {
            "role": "user",
            "content": "Write a story about a CS PhD Student who wants to quit PhD.",
        }
    ],
)

print(response.message.content[0].text)
