import sys 
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))


import torch
from typing import List, Dict, Optional
from dataclasses import dataclass
from pathlib import Path
from src.model.model import Transformer
from src.inference_pipeline.deep_reasoning_inference import TextTokenizer

@dataclass
class AgentConfig:
    role: str
    personality: str
    expertise: List[str]
    temperature: float
    top_p: float

class Agent:
    def __init__(
        self,
        model_path: str,
        config: AgentConfig,
        tokenizer: TextTokenizer
    ):
        self.model = self._load_model(model_path)
        self.config = config
        self.tokenizer = tokenizer
        self.conversation_history = []
        
    def _load_model(self, model_path: str) -> Transformer:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model checkpoint not found at {model_path}")
        
        try:
            checkpoint = torch.load(model_path, map_location='cpu')
        except Exception as e:
            raise RuntimeError(f"Error loading checkpoint: {str(e)}")

        state_dict = checkpoint.get('model_state_dict', checkpoint)
        
        vocab_size, d_model = state_dict['src_embedding.weight'].shape
        
        d_ff = state_dict['encoder_layers.0.feed_forward.linear1.weight'].size(0)
        
        num_layers = sum(1 for key in state_dict if key.startswith('encoder_layers.') and key.endswith('.feed_forward.linear1.weight'))
        
        print(f"Detected model dimensions from checkpoint:")
        print(f"vocab_size: {vocab_size}")
        print(f"d_model: {d_model}")
        print(f"d_ff: {d_ff}")
        print(f"num_layers: {num_layers}")
        
        model_args = {
            'src_vocab_size': vocab_size,
            'tgt_vocab_size': vocab_size,
            'd_model': d_model,
            'num_heads': 8,  # This could also be extracted if needed
            'num_layers': num_layers,
            'd_ff': d_ff,
            'max_len': 5000,
            'dropout': 0.1
        }
        
        model = Transformer(**model_args)
        
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        try:
            model.load_state_dict(state_dict)
        except Exception as e:
            print(f"Warning: Error loading state dict: {str(e)}")
            print("Attempting to load with strict=False...")
            model.load_state_dict(state_dict, strict=False)
        
        model.eval()
        return model
    
    def generate_response(self, prompt: str, context: Optional[str] = None) -> str:
        try:
            formatted_prompt = f"Role: {self.config.role}\nPersonality: {self.config.personality}\n\nContext: {context if context else ''}\n\nPrompt: {prompt}"
            
            input_ids = torch.tensor(self.tokenizer.encode(formatted_prompt)).unsqueeze(0)
            
            print(f"Input shape: {input_ids.shape}")
            
            with torch.no_grad():
                output_ids = self.model(input_ids)  
                
                probs = torch.softmax(output_ids[0, -1] / self.config.temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                generated_ids = [next_token.item()]
                
                max_length = 200
                for _ in range(max_length - 1):
                    current_input = torch.cat([input_ids, torch.tensor([generated_ids]).to(input_ids.device)], dim=1)
                    output = self.model(current_input)
                    
                    probs = torch.softmax(output[0, -1] / self.config.temperature, dim=-1)
                    if self.config.top_p < 1.0:
                        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                        cumsum_probs = torch.cumsum(sorted_probs, dim=-1)
                        sorted_indices_to_remove = cumsum_probs > self.config.top_p
                        sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
                        sorted_indices_to_remove[0] = False
                        probs[sorted_indices[sorted_indices_to_remove]] = 0
                        probs = probs / probs.sum()
                    
                    next_token = torch.multinomial(probs, num_samples=1)
                    generated_ids.append(next_token.item())
                    
                    if next_token.item() == self.tokenizer.tokenizer.token_to_id("</s>"):
                        break
            
            response = self.tokenizer.decode(generated_ids)
            return response
            
        except Exception as e:
            print(f"Error in generate_response: {str(e)}")
            raise
        
        
class MultiAgentSystem:
    def __init__(self, model_path: str, agent_configs: List[AgentConfig]):
        from tokenizers import Tokenizer
        tokenizer_path = str(Path(model_path).parent / "tokenizer.json")
        if not os.path.exists(tokenizer_path):
            raise FileNotFoundError(f"Tokenizer not found at {tokenizer_path}")
            
        print(f"Loading tokenizer from {tokenizer_path}")
        self.base_tokenizer = Tokenizer.from_file(tokenizer_path)
        
        class WrappedTokenizer:
            def __init__(self, tokenizer):
                self.tokenizer = tokenizer
                self.vocab_size = tokenizer.get_vocab_size()
                
            def encode(self, text: str) -> List[int]:
                return self.tokenizer.encode(text).ids
                
            def decode(self, token_ids: List[int]) -> str:
                return self.tokenizer.decode(token_ids)
                
        self.tokenizer = WrappedTokenizer(self.base_tokenizer)
        
        self.agents = [
            Agent(model_path, config, self.tokenizer)
            for config in agent_configs
        ]
        
    def collaborative_response(self, query: str) -> Dict[str, str]:
        """Generate responses from all agents and combine them"""
        responses = {}
        
        for agent in self.agents:
            try:
                print(f"\nGenerating response from {agent.config.role}...")
                response = agent.generate_response(query)
                responses[agent.config.role] = response
                print(f"Response generated successfully.")
            except Exception as e:
                print(f"Error getting response from {agent.config.role}: {str(e)}")
                responses[agent.config.role] = f"Error: {str(e)}"
            
        return responses
    
    def debate(self, topic: str, rounds: int = 3) -> List[Dict[str, str]]:
        """Conduct a multi-round debate between agents"""
        debate_history = []
        
        current_topic = topic
        for _ in range(rounds):
            round_responses = {}
            
            for agent in self.agents:
                response = agent.generate_response(
                    current_topic,
                    context=str(debate_history) if debate_history else None
                )
                round_responses[agent.config.role] = response
            
            debate_history.append(round_responses)
            current_topic = f"Considering the above responses, provide your perspective on {topic}"
            
        return debate_history

if __name__ == "__main__":
    agent_configs = [
        AgentConfig(
            role="Critical Thinker",
            personality="Analytical and detail-oriented",
            expertise=["logic", "analysis"],
            temperature=0.7,
            top_p=0.9
        ),
        AgentConfig(
            role="Creative Explorer",
            personality="Innovative and imaginative",
            expertise=["brainstorming", "lateral thinking"],
            temperature=0.9,
            top_p=0.95
        ),
        AgentConfig(
            role="Pragmatic Implementer",
            personality="Practical and solution-focused",
            expertise=["implementation", "optimization"],
            temperature=0.5,
            top_p=0.8
        )
    ]
    
    model_path = "/home/joseph_woodall/workspace/reasoning_models/output/checkpoints/transformer_lm_latest.pt"
    mas = MultiAgentSystem(model_path, agent_configs)
    
    print("="*100)
    print("\nWelcome to the Multi-Agent System Interface!")
    print("="*100)
    print("\nAvailable modes:")
    print("1. Collaborative Response")
    print("2. Multi-Agent Debate")
    print("="*100)
    
    while True:
        try:
            mode = input("\nSelect mode (1 or 2, or 'q' to quit): ").strip().lower()
            
            if mode == 'q':
                print("\nThank you for using the Multi-Agent System!")
                break
                
            elif mode == '1':
                query = input("\nEnter your question for the agents: ")
                print("\nGenerating collaborative response...")
                responses = mas.collaborative_response(query)
                
                print("\nResponses from agents:")
                print("----------------------")
                for role, response in responses.items():
                    print(f"\n{role}:")
                    print(response)
                    
            elif mode == '2':
                topic = input("\nEnter the debate topic: ")
                rounds = int(input("Enter number of debate rounds (1-5): "))
                rounds = max(1, min(5, rounds))  # Ensure rounds is between 1 and 5
                
                print(f"\nStarting {rounds}-round debate...")
                debate_history = mas.debate(topic, rounds)
                
                print("\nDebate Results:")
                print("--------------")
                for round_num, round_responses in enumerate(debate_history, 1):
                    print(f"\nRound {round_num}:")
                    for role, response in round_responses.items():
                        print(f"\n{role}:")
                        print(response)
                        
            else:
                print("\nInvalid mode. Please select 1 for Collaborative Response or 2 for Debate.")
                
        except KeyboardInterrupt:
            print("\n\nExiting gracefully...")
            break
            
        except Exception as e:
            print(f"\nAn error occurred: {str(e)}")
            print("Please try again.")
