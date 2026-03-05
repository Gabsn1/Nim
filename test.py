from nim import NimAI

def test_get_q_value(ai):
    print("\n--- Testing get_q_value ---")
    state = (0, 0, 0, 2)
    action = (3, 2)
    value = ai.get_q_value(state, action)
    print(f"Q-value for state {state}, action {action}: {value}")


def test_update_q_value(ai):
    print("\n--- Testing update_q_value ---")


def test_best_future_reward(ai):
    print("\n--- Testing best_future_reward ---")


def test_choose_action(ai):
    print("\n--- Testing choose_action ---")


if __name__ == "__main__":
    ai = NimAI()

    test_get_q_value(ai)
    test_update_q_value(ai)
    test_best_future_reward(ai)
    test_choose_action(ai)

    print("\nAll tests completed.")
