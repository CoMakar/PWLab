from typing import List, Callable, Any, Union

def force_input(repeat_msg: str = ">>> ", bad_input_msg: Union[str, None] = None, error_msg: Union[str, None] = None, *,
                func: Callable[[str], Any] = lambda s: s, predicates: List[Callable[[Any], bool]] = []):
    """input function that will force user to enter a correct data

    Args:
        - func (Callable): function to be applied to every user input
        - predicates (List[Callable[[Any], bool]]): list of predicates that the user data must match (List of function returning Boolean)
        user data must match all the predicates
        - bad_input_msg (str): message to be if user input does not match all the predicates
        - repeat_msg (str): message to be displayed every iteration
        - error_msg (str, optional): message to be displayed if error occurs in `func` or `predicates`
    """
    
    user_input = None
    while user_input is None:
        try:
            user_input = input(repeat_msg)
            
            user_input = func(user_input)
            
            if predicates:
                if not all([condition(user_input) for condition in predicates]):
                    user_input = None
                    
                    if bad_input_msg:
                        print(bad_input_msg)
            
        except Exception as e:
            user_input = None
            if error_msg:
                print(error_msg)
    
    return user_input