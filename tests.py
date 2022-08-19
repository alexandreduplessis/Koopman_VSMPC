# from src.env.vs_env import VsEnv


# def function(env):
#     env.reset(x=1., y=2., depth=0.3)
#     env.render()
#     for _ in range(100):
#         env.step({"abscisse": 0, "ordonnee": 0, "depth": 0})
#         env.render()
#         print(env.state)
#         print(env.reward)
#         print(env.done)
#         print()
#         if env.done:
#             break

# if __name__ == "__main__":
#     env = VsEnv(length=10, width=10, goal=[5, 5, 1], initial=[1, 1, 1], max_steps=100)
#     function(env)

import logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(levelname)-8s :: %(name)s [ %(lineno)s ]  :: %(message)s \r"
)

# --- Log to Stream
logger = logging.getLogger('scratch')
logger.debug("Test_1\r")
logger.debug('Test_2')
logger.debug('Test_3')
test_dict = {'message_type': 'update',
             'date': '03-07-2020',
             'params':
                 {'p_1': 'X',
                  'p_2': 'Y',
                  'p_3': 'Z'}
             }