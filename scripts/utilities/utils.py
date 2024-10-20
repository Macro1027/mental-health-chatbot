def get_emotions():
    with open("data/external/emotions.txt", "r") as data:
        emotions = data.read().replace("/", "\n").strip().replace("\n\n", "\n")
        return emotions.split()