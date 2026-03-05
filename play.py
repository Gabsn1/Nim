from nim import train
from game import start_game

if __name__ == "__main__":
    # Train the AI with 1000 games
    print("START TRAINING \n")
    ai = train(1000)

    # Start the game and play against the trained AI
    print("STARTING THE GAME \n")
    start_game(ai)
