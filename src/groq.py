from dotenv import load_dotenv

from groq import Groq

import os


def get_groq_completions(user_content):
    _ = load_dotenv()

    client = Groq(
        api_key=os.environ['GROQ_API_KEY'],
    )

    completion = client.chat.completions.create(
        model="mixtral-8x7b-32768",
        messages=[
            {
                "role": "system",
                "content": "Tu es un journaliste senior qui aime rétablir la vérité dans les informations. \nRépond à la question en français et dit que tu ne sais pas si tu n'as pas l'information."
            },
            {
                "role": "user",
                "content": user_content
            }
        ],
        temperature=0.5,
        max_tokens=5640,
        top_p=1,
        stream=True,
        stop=None,
    )

    result = ""
    for chunk in completion:
        result += chunk.choices[0].delta.content or ""

    return result
