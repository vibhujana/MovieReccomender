
from reccomender import *

def __main__():
    r = Reccomender(False)
    users = Reccomender.get_user_data()
    user_input = ""
    user = None
    while(user_input != 'quit'):
        if user == None:
            user_input = input("Press 1 to sign in. Press 2 to sign up. Type quit to quit: ")
            if(user_input == "1"):
                username = input("Type a username: ") 
                user = User.get_user(users,username)
                if type(user.userInfo) != dict:
                    username = None
                    continue
            elif(user_input == "2"):
                username = input("Type a username: ") 
                user = User.create_user(users,username)
            elif(user_input == "quit"):
                continue
        print("Welcome " + user.key + "!")
        user_input = input("What do you want to do: search and add a movie(1), see movies i like(2), like a reccomended movie(3), reccomend titles(4), delete a movie i like (5) or logout(6)")
        if user_input == "1":
            movie_name = input("Type the movie you like: ")
            closest_names = r.search_for_title(movie_name)
            display(closest_names.to_string())
            movie_id = input("Type the id of the title you like: ")
            user = r.add_title(user,movie_id)
        if user_input == "2":
            print(user.key)
            print(user.userInfo[user.key]["liked_media"])
            r.display_movies_i_like(user)
        if user_input == "3": #not working
            r.display_movies_i_got_rec(user)
            movie_id = input("Type the tconst of the movie you got reccomended and liked: ")
            if(r.isValidId(movie_id)):
                print("valid")
                user.like_rec(movie_id)
        if user_input == "4": 
            count = int(input("How many movies to reccomend? "))
            user_vector = r.calc_avg_like(user)
            sim_vectors = r.similarity_to_avg(user_vector,count,user)
            display(r.get_data_of_rec_titles(sim_vectors,user))
        if user_input == '5':
            r.display_movies_i_like(user)
            movie_id = input("Type tconst of what you want to remove")
            user.delete_liked_movie(movie_id)
        if user_input == "6": #not working
            user = Reccomender.logout(user,users)
    Reccomender.save_users(users)
__main__()

