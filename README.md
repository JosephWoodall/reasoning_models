the agent is the llm itself 
the action is the python interpreter 
the reward is 1 for a correct response and 0 for a incorrect response 

![How to apply RL to LLMs](screenshots/Screenshot_20250811_150257.png)
![How to apply RL to LLMs 2] (screenshots/Screenshot_20250811_150409.png)
![How to apply RL to LLMs 3] (screenshots/Screenshot_20250811_151058.png)
![How to apply RL to LLMs 3] (screenshots/Screenshot_20250811_151208.png)
The state of the current iteration i are the tokens completed by the LLM up until i and action is the next token or next word that the LLM is predicting
s0 is always the prompt 
a0 is the subject of the prompt 
sN is always the aN of the previous state 
Likewise, aN is always what the LLM receives in the prompt for the next iteration