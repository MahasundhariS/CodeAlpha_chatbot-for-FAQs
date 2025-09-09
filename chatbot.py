# chatbot.py
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Step 1: Define FAQ data (question-answer pairs)
faqs = {
    "What is AI?": "AI stands for Artificial Intelligence. It is the simulation of human intelligence in machines.",
    "What is Python?": "Python is a popular programming language for AI, data science, and web development.",
    "What is CodeAlpha?": "CodeAlpha is an internship and training platform.",
    "How can I apply for CodeAlpha internship?": "You can apply for a CodeAlpha internship through their official website or LinkedIn posts.",
    "What is Machine Learning?": "ML is a subset of AI that learns from data.",
    "What is Deep Learning?": "Deep Learning uses neural networks for advanced AI tasks.",
}

# Step 2: Prepare data
questions = list(faqs.keys())
answers = list(faqs.values())

# Step 3: Convert text to vectors using TF-IDF
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(questions)

# Step 4: Define chatbot response function
def chatbot(user_input):
    user_vec = vectorizer.transform([user_input])
    similarity = cosine_similarity(user_vec, X)
    best_match_index = similarity.argmax()
    best_score = similarity[0][best_match_index]

    if best_score > 0.2:  # threshold to avoid wrong matches
        return answers[best_match_index]
    else:
        return "Sorry, I don't know the answer. Please try rephrasing."

# Step 5: Run chatbot in console
if __name__== "__main__":
    print("🤖 Chatbot: Hello! Ask me something (type 'exit' to quit)")
    while True:
        user = input("You: ")
        if user.lower() == "exit":
            print("🤖 Chatbot: Goodbye!")
            break
        print("🤖 Chatbot:", chatbot(user))
