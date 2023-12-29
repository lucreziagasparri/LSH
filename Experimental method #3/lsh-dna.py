from datasketch import MinHashLSH, MinHash
import json
import time

threshold = 0.3
num_perm = 128    
shingle_length = 3

def generate_dna(tweets):
    dna = ''
    for tweet_data in tweets:
        if tweet_data.startswith("RT @"):
            dna += "C"
        elif "@" in tweet_data:
            dna += "T"
        elif 'http' in tweet_data:
            dna += "U"
        elif '#' in tweet_data:
            dna += "H"
        else:
            dna += "A"
    return dna

def generate_shingles(sequence, shingle_length):
    shingles = set()
    for i in range(len(sequence) - shingle_length + 1):
        shingle = sequence[i:i + shingle_length]
        shingles.add(shingle)
    return shingles

def generate_minhash(shingles, num_perm):
    minhash = MinHash(num_perm=num_perm)
    for shingle in shingles:
        minhash.update(shingle.encode('utf-8'))
    return minhash

def jaccard_similarity_calculate(minhash1, minhash2):
    return minhash1.jaccard(minhash2)

def label_calculate(similarities):     
    sorted_first5_similarities = sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:5]
    sum_bot = 0
    sum_human = 0
    for user, similarity in sorted_first5_similarities:
        if user in users:
            user_category = users[user]['Categoria']
            print(f"Utente: {user}, Similarità di Jaccard: {similarity}, Categoria: {user_category}")
            if user_category == "Bot":
                sum_bot = sum_bot + similarity
            elif user_category == "Umano":
                sum_human = sum_human + similarity

    if(sum_bot > sum_human):
        print("Calcolato come BOT")

        if(query_user_info['Categoria'] == "Bot"):
            print("TRUE POSITIVE - BOT")

        elif(query_user_info['Categoria'] == "Umano"):
            print("FALSE POSITIVE - BOT")

    elif(sum_human > sum_bot):
        print("Calcolato come UMANO")

        if(query_user_info['Categoria'] == "Umano"):
            print("TRUE POSITIVE - UMANO")

        elif(query_user_info['Categoria'] == "Bot"):
            print("FALSE POSITIVE - UMANO")
    else:
        print("Non sono in grado di calcolarlo")




start_time = time.time()

# File JSON di ground truth
with open('union_400_ground_truth.json', 'r') as json_file:
    user_data = json.load(json_file)

users = {}
for user_entry in user_data:
    user_id = user_entry['User ID']
    tweets = [(tweet["Testo"]) for tweet in user_entry["Tweets"]]
    user_info = {
        "Categoria": user_entry["Categoria"],
        "Tweets": tweets
    }
    users[user_id] = user_info

lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)

minhashes = {}
for user_id, user_info in users.items():
    dna_sequence = generate_dna(user_info["Tweets"])
    shingles = generate_shingles(dna_sequence, shingle_length=shingle_length)
    minhash = generate_minhash(shingles, num_perm=num_perm)
    lsh.insert(user_id, minhash)
    minhashes[user_id] = minhash 


with open('union_1600_test.json', 'r') as query_file:
    query_data_list = json.load(query_file)

for query_data in query_data_list:
    query_user = query_data["User ID"]
    query_tweets = [(tweet["Testo"]) for tweet in query_data["Tweets"]]
    query_user_info = {
        "Categoria": query_data["Categoria"],
        "Tweets": query_tweets
    }
    
    print("\n-------------------------------------------------------------------------------------------------------------------")
    print(f"Dato di query: {query_user}")
    print(f"Categoria: {query_user_info['Categoria']}")

    query_sequence = generate_dna(query_user_info["Tweets"])
    print("QUERY DNA", query_sequence)
    query_shingles = generate_shingles(query_sequence, shingle_length=shingle_length)
    print("QUERY SHINGLES", query_shingles)
    query_minhash = generate_minhash(query_shingles, num_perm=num_perm)

    
    result = lsh.query(query_minhash)
    print("Sequenze simili trovate:", result)
    print("Totale sequenze simili trovate:", len(result))

    category_bot = {}
    category_human = {}
    for user_id in result:
        if user_id in users:
            category = users[user_id]["Categoria"]
            if category == "Bot":
                category_bot[user_id] = users[user_id]
            elif category == "Umano":
                category_human[user_id] = users[user_id]

    print(f"Bot: {len(category_bot)}")
    print(f"Umano: {len(category_human)}\n")

    if len(result) > 0:
        similarities = {}
        for user, minhash in minhashes.items():
            if user in result:
                jaccard_coefficient = jaccard_similarity_calculate(query_minhash, minhash)
                similarities[user] = jaccard_coefficient
        label_calculate(similarities)
    else:
        print("Non riesco a determinare risultati simili per questo record")



end_time = time.time()

execution_time = end_time - start_time

print(f"Il tempo di esecuzione è stato di {execution_time} secondi")