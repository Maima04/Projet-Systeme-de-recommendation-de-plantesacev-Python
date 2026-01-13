import tkinter as tk
from tkinter import Canvas, Scrollbar, Frame, Label, Button, Entry, Toplevel, messagebox
from PIL import Image, ImageTk, ImageOps
import mysql.connector
import numpy as np
from scipy.spatial import distance
from sklearn.feature_extraction.text import TfidfVectorizer
import io
from functools import partial
from collections import Counter

# Fonctions de base pour la connexion à la base de données et la récupération des données
def get_connection():
    return mysql.connector.connect(
        host="localhost",
        port=3306,
        user="root",
        password="",
        database="python" 
    )

def fetch_data():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id, produit, description, photo, prix FROM produit")
    produits = cursor.fetchall()
    cursor.execute("SELECT count(*) FROM user")
    nb_user = int(cursor.fetchone()[0])
    matrice_user_item = np.zeros((nb_user, len(produits)))
    cursor.execute("SELECT iduser, id, note FROM note")
    for row in cursor:
        user_id = row[0] - 1  # Adjust for 0-based indexing
        product_id = row[1] - 1  # Adjust for 0-based indexing
        if 0 <= user_id < nb_user and 0 <= product_id < len(produits):
            matrice_user_item[user_id][product_id] = row[2]  # Assign rating
    # Fetch average ratings for each product
    cursor.execute("SELECT id, AVG(note) as avg_rating FROM note GROUP BY id")
    avg_ratings = {row[0]: round(row[1], 2) for row in cursor}  # Store as {product_id: avg_rating}
    cursor.close()
    conn.close()
    return produits, matrice_user_item, nb_user, avg_ratings

produits, matrice_user_item, nb_user, avg_ratings = fetch_data()

def similarite_cosinus(a, b):
    if np.count_nonzero(matrice_user_item[a]) == 0 or np.count_nonzero(matrice_user_item[b]) == 0:
        return 0
    return 1 - distance.cosine(matrice_user_item[a], matrice_user_item[b])

def compute_user_user_matrix():
    matrice_user_user = np.zeros((nb_user, nb_user))
    for i in range(nb_user):
        for j in range(nb_user):
            if i != j:
                matrice_user_user[i][j] = similarite_cosinus(i, j)
            else:
                matrice_user_user[i][j] = 0
    return matrice_user_user

matrice_user_user = compute_user_user_matrix()

def prepare_data(produits):
    texts = [p[2] for p in produits]
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(texts)
    return tfidf_matrix, {p[0]: (p[1], p[3], p[4]) for p in produits}  # Include prix

tfidf_matrix, product_dict = prepare_data(produits)
cosine_similarities = 1 - distance.squareform(distance.pdist(tfidf_matrix.toarray(), 'cosine'))
product_ids = list(product_dict.keys())

root = tk.Tk()
root.title("La Canopée")
root.geometry("1200x800")

header = Label(root, text="La Canopée", font=('Helvetica', 20, 'bold'), bg='#c7deaa', fg='#000000')
header.pack(side=tk.TOP, fill=tk.X)

user_id_label = Label(root, text="Entrez votre ID utilisateur:", fg='#000000')
user_id_label.pack()

user_id_entry = Entry(root, bg='WHITE', fg='#000000', insertbackground='#000000')
user_id_entry.pack()

# Adding a decorative element
decor_frame = Frame(root, bg='#c7deaa', height=40)
decor_frame.pack(fill=tk.X, pady=10)
Label(decor_frame, text="✿ Explore Our Collection ✿", font=('Helvetica', 16, 'italic'), bg='#c7deaa', fg='#FFFFFF').pack(pady=5)

canvas = Canvas(root, bg='#f3ede0')
canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
scroll_y = Scrollbar(root, orient="vertical", command=canvas.yview)
scroll_y.pack(side=tk.RIGHT, fill=tk.Y)
canvas.configure(yscrollcommand=scroll_y.set)
scrollable_frame = Frame(canvas, bg='#f3ede0')
canvas.create_window((0, 0), window=scrollable_frame, anchor='nw')

images = []

def add_rating(product_id):
    rating_window = Toplevel(root)
    rating_window.title("Noter le produit")
    rating_window.geometry("300x200")
    
    Label(rating_window, text=f"Noter {product_dict[product_id][0]} (1-5):", font=('Helvetica', 12)).pack(pady=10)
    
    rating_entry = Entry(rating_window)
    rating_entry.pack(pady=10)
    
    def submit_rating():
        try:
            user_id = int(user_id_entry.get())
            rating = int(rating_entry.get())
            if not (1 <= user_id <= nb_user):
                messagebox.showerror("Erreur", f"L'ID utilisateur doit être entre 1 et {nb_user}.")
                return
            if not (1 <= rating <= 5):
                messagebox.showerror("Erreur", "La note doit être entre 1 et 5.")
                return
            
            # Save rating to database
            conn = get_connection()
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO note (iduser, id, note) 
                VALUES (%s, %s, %s) 
                ON DUPLICATE KEY UPDATE note = %s
            """, (user_id, product_id, rating, rating))
            conn.commit()
            
            # Update user-item matrix
            global matrice_user_item, matrice_user_user, avg_ratings
            matrice_user_item[user_id - 1][product_id - 1] = rating
            matrice_user_user = compute_user_user_matrix()
            
            # Update average rating for the product
            cursor.execute("SELECT AVG(note) FROM note WHERE id = %s", (product_id,))
            result = cursor.fetchone()
            avg_ratings[product_id] = round(result[0], 2) if result[0] is not None else 0.0
            
            cursor.close()
            conn.close()
            messagebox.showinfo("Succès", "Note enregistrée avec succès!")
            rating_window.destroy()
            display_all_products()  # Refresh display to show updated average rating
            
        except ValueError:
            messagebox.showerror("Erreur", "Veuillez entrer un ID utilisateur et une note valides (nombres entiers).")
        except mysql.connector.Error as e:
            messagebox.showerror("Erreur", f"Erreur de base de données: {e}")
    
    submit_button = Button(rating_window, text="Soumettre", command=submit_rating, bg='#d7d9c6', fg='#000000')
    submit_button.pack(pady=10)

def display_product(product_id, idx):
    name, photo_data, price = product_dict[product_id]
    if not photo_data:
        return
    try:
        image = Image.open(io.BytesIO(photo_data))
        image = ImageOps.contain(image, (150, 150))
        photo = ImageTk.PhotoImage(image)
        images.append(photo)

        # Create a Canvas for less sharp square frame with rounded corners
        frame_canvas = Canvas(scrollable_frame, bg='#E5D7C4', highlightthickness=2, highlightbackground='#E5D7C4', width=180, height=300)  # Increased height for avg rating
        frame_canvas.grid(row=idx // 5, column=idx % 5, padx=20, pady=20)

        # Draw a rectangle with softened corners (approximated)
        frame_canvas.create_rectangle(10, 10, 170, 290, fill='#FFF5F7', outline='#E5D7C4', width=2)

        # Place image inside the frame
        panel = Label(frame_canvas, image=photo, bg='#FFF5F7')
        panel.image = photo
        panel.place(x=15, y=15, width=150, height=150)

        # Place label inside the frame
        label = Label(frame_canvas, text=name, bg='#FFF5F7', fg='#000000', font=('Helvetica', 12, 'bold'), wraplength=140)
        label.place(x=20, y=170)

        # Place average rating label inside the frame
        avg_rating = avg_ratings.get(product_id, 0.0)
        avg_rating_label = Label(frame_canvas, text=f"Note moyenne: {avg_rating}/5", bg='#FFF5F7', fg='#000000', font=('Helvetica', 10))
        avg_rating_label.place(x=20, y=195)

        # Place price label inside the frame
        price_label = Label(frame_canvas, text=f"Price: {price}.00 DT", bg='#FFF5F7', fg='#000000', font=('Helvetica', 10))
        price_label.place(x=20, y=220)

        # Place "Recommander" button inside the frame
        btn_recommend = Button(frame_canvas, text="Recommander", command=partial(show_recommendations, product_id), bg='#d7d9c6', fg='#000000')
        btn_recommend.place(x=20, y=245, width=80)

        # Place "Note" button inside the frame
        btn_rate = Button(frame_canvas, text="Note", command=partial(add_rating, product_id), bg='#d7d9c6', fg='#000000')
        btn_rate.place(x=110, y=245, width=50)

    except Exception as e:
        print(f"Error loading product {name}: {e}")

def show_recommendations(product_id):
    top_window = Toplevel(root)
    top_window.title("Recommandations pour " + product_dict[product_id][0])
    top_window.geometry("600x400")
    content_label = Label(top_window, text="Recommandations basées sur le contenu :", font=('Helvetica', 12, 'bold'), fg='#000000')
    content_label.pack()

    recommendations = get_recommendations(product_id)
    for idx, (rec_id, sim) in enumerate(recommendations):
        rec_name = product_dict[rec_id][0]
        Label(top_window, text=f"{rec_name} - Similarité: {sim:.2f}", fg='#000000').pack()

    collab_label = Label(top_window, text="Recommandations collaboratives :", font=('Helvetica', 12, 'bold'), fg='#000000')
    collab_label.pack()

    try:
        user_id_input = user_id_entry.get()
        if not user_id_input:
            Label(top_window, text="Erreur : Veuillez entrer un ID utilisateur.", fg='#000000').pack()
            return
        user_id = int(user_id_input)
        if 1 <= user_id <= nb_user:
            similarites = list(enumerate(matrice_user_user[user_id - 1]))
            similarites.sort(key=lambda x: x[1], reverse=True)
            top_3 = similarites[1:4]
            recommandations = []
            for idx, (sim_idx, sim) in enumerate(top_3):
                user_items = np.where(matrice_user_item[sim_idx] > 0)[0] + 1  # Correction pour l'indexation
                recommandations.extend(user_items)

            recommandations = [item for item, freq in Counter(recommandations).most_common(3)]
            for rec_id in recommandations:
                rec_name = product_dict[rec_id][0]
                Label(top_window, text=rec_name, fg='#000000').pack()
        else:
            Label(top_window, text=f"Erreur : L'ID utilisateur doit être entre 1 et {nb_user}.", fg='#000000').pack()
    except ValueError:
        Label(top_window, text="Erreur : Veuillez entrer un ID utilisateur valide (nombre entier).", fg='#000000').pack()

def get_recommendations(product_id):
    index = product_ids.index(product_id)
    similarities = cosine_similarities[index]
    sorted_indices = np.argsort(similarities)[::-1][1:4]
    recommendations = [(product_ids[i], similarities[i]) for i in sorted_indices]
    return recommendations

def clear_display():
    for widget in scrollable_frame.winfo_children():
        widget.destroy()

def display_all_products():
    clear_display()
    for idx, product_id in enumerate(product_ids):
        display_product(product_id, idx)

display_all_products()

refresh_button = Button(root, text="Actualiser", command=display_all_products, fg='#000000')
refresh_button.pack(side=tk.TOP, pady=10)

def update_scrollregion():
    canvas.configure(scrollregion=canvas.bbox("all"))

update_scrollregion()

root.mainloop()
message.txt