import wikipedia


def search_wikipedia(query):
    try:
        wikipedia.set_lang('fr')
        search_results = wikipedia.search(query)
        if not search_results:
            return "Aucun résultat trouvé sur Wikipédia pour cette requête."

        page_title = search_results[0]
        page = wikipedia.page(page_title)

        summary = page.summary
        return summary
    except wikipedia.exceptions.DisambiguationError as e:
        return "La requête correspond à plusieurs résultats. Veuillez préciser."


def ask_question(question):
    return search_wikipedia(question)


if __name__ == "__main__":
    # test utilisation:
    user_question = input("Posez votre question : ")
    result = ask_question(user_question)
    print(result)