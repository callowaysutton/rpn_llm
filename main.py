import random
import operator
import time
from typing import Tuple, List, Dict, Any
import json
import numpy as np
import ollama
from tqdm import tqdm
import matplotlib.pyplot as plt

# Dictionary of operators with their symbols, precedence, and function implementations
OPERATORS = {
    '+': {'precedence': 1, 'func': operator.add, 'assoc': 'left'},
    '-': {'precedence': 1, 'func': operator.sub, 'assoc': 'left'},
    '*': {'precedence': 2, 'func': operator.mul, 'assoc': 'left'},
    '/': {'precedence': 2, 'func': operator.truediv, 'assoc': 'left'},
    '^': {'precedence': 3, 'func': operator.pow, 'assoc': 'right'}
}

class EquationGenerator:
    """Generates equations in both RPN and infix notation"""
    
    def __init__(self, max_depth: int = 3, max_value: int = 20, parentheses_prob: float = 0.4, max_result: int = 1000000, min_result: int = -1000000):
        """
        Initialize the EquationGenerator
        
        Args:
            max_depth (int): Maximum depth of the expression tree
            max_value (int): Maximum value for numbers in the equation
            parentheses_prob (float): Probability of adding explicit parentheses
            max_result (int): Maximum result value to avoid overflow
        """
        self.max_depth = max_depth
        self.max_value = max_value
        self.max_result = max_result
        self.min_result = min_result
        self.parentheses_prob = parentheses_prob  # Probability of adding explicit parentheses
        
    def generate_equation(self) -> Tuple[str, str, float]:
        """
        Generate an equation and return it in both notations along with the answer
        
        Returns:
            Tuple[str, str, float]: (infix_notation, rpn_notation, result)
        """
        # Generate an expression tree
        expression_tree = self._generate_expression_tree(depth=0)
        
        # Convert to notations
        infix = self._tree_to_infix(expression_tree)
        rpn = self._tree_to_rpn(expression_tree)
        
        # Compute the result
        result = self._evaluate_tree(expression_tree)
        
        # Check if the result is greater than the max value or smaller than the min value
        if result > self.max_result or result < self.min_result:
            return self.generate_equation()
        
        # If result appears in the infix notation as a standalone number, regenerate
        result_str = f"{result:.2f}".rstrip('0').rstrip('.')
        tokens = infix.split()
        if result_str in tokens:
            return self.generate_equation()
        
        return infix, rpn, result
    
    def generate_complex_equation(self) -> Tuple[str, str, float]:
        """
        Generate a more complex equation that specifically challenges PEMDAS understanding
        """
        # Create patterns that are likely to cause PEMDAS confusion
        patterns = [
            # Pattern 1: a + b * c
            self._create_mixed_precedence_equation,
            # Pattern 2: a * (b + c) - d
            self._create_parenthesized_equation,
            # Pattern 3: a + b - c * d / e
            self._create_multi_operator_equation,
            # Pattern 4: a^b * c + d
            self._create_exponent_equation
        ]
        
        # Choose a random pattern
        pattern_func = random.choice(patterns)
        tree = pattern_func()
        
        # Convert to notations
        infix = self._tree_to_infix(tree)
        rpn = self._tree_to_rpn(tree)
        
        # Compute the result
        result = self._evaluate_tree(tree)
        
        # Check if the result is greater than the max value or smaller than the min value
        if result > self.max_result or result < self.min_result:
            return self.generate_equation()
        
        # Check if the result appears in the infix notation as a standalone number
        result_str = f"{result:.2f}".rstrip('0').rstrip('.')
        tokens = infix.split()
        if result_str in tokens:
            return self.generate_complex_equation()
        
        return infix, rpn, result
    
    def _create_mixed_precedence_equation(self):
        """Create an equation like 'a + b * c' to test precedence understanding"""
        a = {'type': 'number', 'value': random.randint(1, self.max_value)}
        b = {'type': 'number', 'value': random.randint(1, self.max_value)}
        c = {'type': 'number', 'value': random.randint(1, self.max_value)}
        
        # b * c
        mul_expr = {
            'type': 'operation',
            'operator': '*',
            'left': b,
            'right': c
        }
        
        # a + (b * c)
        return {
            'type': 'operation',
            'operator': '+',
            'left': a,
            'right': mul_expr
        }
    
    def _create_parenthesized_equation(self):
        """Create an equation like 'a * (b + c) - d' to test parentheses understanding"""
        a = {'type': 'number', 'value': random.randint(1, self.max_value)}
        b = {'type': 'number', 'value': random.randint(1, self.max_value)}
        c = {'type': 'number', 'value': random.randint(1, self.max_value)}
        d = {'type': 'number', 'value': random.randint(1, self.max_value)}
        
        # b + c
        add_expr = {
            'type': 'operation',
            'operator': '+',
            'left': b,
            'right': c
        }
        
        # a * (b + c)
        mul_expr = {
            'type': 'operation',
            'operator': '*',
            'left': a,
            'right': add_expr
        }
        
        # a * (b + c) - d
        return {
            'type': 'operation',
            'operator': '-',
            'left': mul_expr,
            'right': d
        }
    
    def _create_multi_operator_equation(self):
        """Create an equation with multiple operators like 'a + b - c * d / e'"""
        a = {'type': 'number', 'value': random.randint(1, self.max_value)}
        b = {'type': 'number', 'value': random.randint(1, self.max_value)}
        c = {'type': 'number', 'value': random.randint(1, self.max_value)}
        d = {'type': 'number', 'value': random.randint(1, self.max_value)}
        e = {'type': 'number', 'value': random.randint(1, self.max_value) + 1}  # +1 to avoid division by zero
        
        # d / e
        div_expr = {
            'type': 'operation',
            'operator': '/',
            'left': d,
            'right': e
        }
        
        # c * (d / e)
        mul_expr = {
            'type': 'operation',
            'operator': '*',
            'left': c,
            'right': div_expr
        }
        
        # b - (c * (d / e))
        sub_expr = {
            'type': 'operation',
            'operator': '-',
            'left': b,
            'right': mul_expr
        }
        
        # a + (b - (c * (d / e)))
        return {
            'type': 'operation',
            'operator': '+',
            'left': a,
            'right': sub_expr
        }
    
    def _create_exponent_equation(self):
        """Create an equation with exponents like 'a^b * c + d'"""
        a = {'type': 'number', 'value': random.randint(1, 5)}  # Smaller base for exponents
        b = {'type': 'number', 'value': random.randint(1, 3)}  # Smaller exponent
        c = {'type': 'number', 'value': random.randint(1, self.max_value)}
        d = {'type': 'number', 'value': random.randint(1, self.max_value)}
        
        # a^b
        pow_expr = {
            'type': 'operation',
            'operator': '^',
            'left': a,
            'right': b
        }
        
        # (a^b) * c
        mul_expr = {
            'type': 'operation',
            'operator': '*',
            'left': pow_expr,
            'right': c
        }
        
        # (a^b * c) + d
        return {
            'type': 'operation',
            'operator': '+',
            'left': mul_expr,
            'right': d
        }
    
    def _generate_expression_tree(self, depth: int = 0) -> Dict[str, Any]:
        """Generate a random expression tree with the specified depth"""
        # At maximum depth or with 30% chance at any depth, generate a leaf (number)
        if depth >= self.max_depth or (depth > 0 and random.random() < 0.3):
            return {
                'type': 'number',
                'value': random.randint(1, self.max_value)
            }
        
        # Otherwise, generate an operation node
        op = random.choice(list(OPERATORS.keys()))
        
        return {
            'type': 'operation',
            'operator': op,
            'left': self._generate_expression_tree(depth + 1),
            'right': self._generate_expression_tree(depth + 1)
        }
    
    def _tree_to_infix(self, tree: Dict[str, Any], parent_precedence: int = 0) -> str:
        """Convert expression tree to infix notation with proper parentheses"""
        if tree['type'] == 'number':
            return str(tree['value'])
        
        op = tree['operator']
        op_precedence = OPERATORS[op]['precedence']
        
        left_str = self._tree_to_infix(tree['left'], op_precedence)
        right_str = self._tree_to_infix(tree['right'], op_precedence)
        
        # Add parentheses when the parent operator has higher precedence
        # or when operators have equal precedence but are left-associative
        needs_parens = (op_precedence < parent_precedence or 
                       (op_precedence == parent_precedence and 
                        OPERATORS[op]['assoc'] == 'left'))
        
        expr = f"{left_str} {op} {right_str}"
        
        if needs_parens:
            return f"({expr})"
        return expr
    
    def _tree_to_rpn(self, tree: Dict[str, Any]) -> str:
        """Convert expression tree to Reverse Polish Notation"""
        if tree['type'] == 'number':
            return str(tree['value'])
        
        left_str = self._tree_to_rpn(tree['left'])
        right_str = self._tree_to_rpn(tree['right'])
        
        return f"{left_str} {right_str} {tree['operator']}"
    
    def _evaluate_tree(self, tree: Dict[str, Any]) -> float:
        """Evaluate the expression tree to get the result"""
        if tree['type'] == 'number':
            return tree['value']
        
        left_val = self._evaluate_tree(tree['left'])
        right_val = self._evaluate_tree(tree['right'])
        
        return OPERATORS[tree['operator']]['func'](left_val, right_val)


class ModelTester:
    """Test LLM performance on mathematical equations in different notations"""
    
    def __init__(self, model_name: str = "llama3.2"):
        self.model_name = model_name
        
    def query_model(self, prompt: str, use_grammar: bool = False) -> str:
        """Query the Ollama model with a prompt"""
        try:
            response = ollama.chat(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                # Note: Grammar support would go here if enabled in Ollama
            )
            return response['message']['content']
        except Exception as e:
            print(f"Error querying model: {e}")
            return ""
    
    def is_answer_correct(self, response: str, expected: float) -> bool:
        """
        Check if the expected answer is present in the response
        
        Args:
            response: The model's response text
            expected: The expected answer
            
        Returns:
            bool: True if the expected answer is found in the response
        """
        # Generate different possible string representations of the answer
        # to account for different formatting the model might use
        possible_formats = [
            f"{expected:.2f}".rstrip('0').rstrip('.'),  # 2 decimal places, trimmed
            f"{expected:.1f}".rstrip('0').rstrip('.'),  # 1 decimal place, trimmed
            # f"{int(expected)}" if expected else None,  # Integer without decimal
            # f"{expected:.0f}" if expected else None,  # Integer format
            f"{expected:g}",  # General format (removes trailing zeros)
        ]
                
        # Filter out None values
        possible_formats = [fmt for fmt in possible_formats if fmt is not None]
        
        # Remove any commas from the response
        response = response.replace(",", "")
        
        # Check if any of the possible formats appear in the response
        for fmt in possible_formats:
            if fmt in response:
                return True
                
        return False
    
    def test_equation(self, infix: str, rpn: str, expected: float) -> Dict[str, Any]:
        """Test an equation in both notations and return results"""
        infix_prompt = f"You are an arithmetic math expert. Calculate the result of this mathematical expression: '{infix}'"
        rpn_prompt = f"You are an arithmetic math expert. Calculate the result of this mathematical expression which uses Reverse Polish Notation: '{rpn}'"
        
        # Query the model with both notations
        infix_response = self.query_model(infix_prompt)
        rpn_response = self.query_model(rpn_prompt)
        
        # Check if the responses contain the correct answer
        infix_correct = self.is_answer_correct(infix_response, expected)
        rpn_correct = self.is_answer_correct(rpn_response, expected)
        
        return {
            "infix": infix,
            "rpn": rpn,
            "expected": expected,
            "rpn_prompt": rpn_prompt,
            "rpn_response": rpn_response,
            "infix_prompt": infix_prompt,
            "infix_response": infix_response,
            "infix_correct": infix_correct,
            "rpn_correct": rpn_correct,
        }


def run_experiment(num_equations: int = 1000):
    """Run the full experiment comparing RPN vs Infix notation"""
    generator = EquationGenerator(max_depth=1, max_value=10)
    tester = ModelTester()
    
    results = []
    
    print(f"Starting experiment with {num_equations} equations...")
    
    for i in tqdm(range(num_equations)):
        try:
            # Generate an equation
            infix, rpn, expected = generator.generate_equation()
            
            # Test the equation
            result = tester.test_equation(infix, rpn, expected)
            results.append(result)
            
            # Sleep a bit to avoid overwhelming Ollama
            time.sleep(0.5)
            
        except Exception as e:
            print(f"Error on equation {i}: {e}")
    
    # Analyze results
    analyze_and_visualize_results(results)
    
    # Save results to file
    with open("results.json", "w") as f:
        json.dump(results, f, indent=4)
    
    return results


def analyze_and_visualize_results(results: List[Dict[str, Any]]):
    """Analyze and visualize the experiment results"""
    infix_correct = sum(r['infix_correct'] for r in results)
    rpn_correct = sum(r['rpn_correct'] for r in results)
    total = len(results)
    
    print(f"\nResults Summary:")
    print(f"Total equations tested: {total}")
    print(f"Infix notation correct: {infix_correct} ({infix_correct/total*100:.2f}%)")
    print(f"RPN notation correct: {rpn_correct} ({rpn_correct/total*100:.2f}%)")
    
    # Create a bar chart
    labels = ['Infix Notation', 'RPN Notation']
    correct_counts = [infix_correct, rpn_correct]
    incorrect_counts = [total - infix_correct, total - rpn_correct]
    
    x = np.arange(len(labels))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, correct_counts, width, label='Correct')
    rects2 = ax.bar(x + width/2, incorrect_counts, width, label='Incorrect')
    
    ax.set_title('Model Performance by Notation Type')
    ax.set_xlabel('Notation Type')
    ax.set_ylabel('Number of Equations')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    
    # Add percentages on top of bars
    for rect in rects1:
        height = rect.get_height()
        ax.annotate(f'{height/total*100:.1f}%',
                    xy=(rect.get_x() + rect.get_width()/2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('results_comparison.png')
    plt.close()


if __name__ == "__main__":
    # For testing a single equation
    generator = EquationGenerator()
    infix, rpn, result = generator.generate_equation()
    print(f"Infix: {infix}")
    print(f"RPN: {rpn}")
    print(f"Expected result: {result}")
    
    # Ask if the user wants to run the full experiment
    response = input("\nDo you want to run the full experiment with 1000 equations? (y/n): ")
    if response.lower() == 'y':
        num_equations = int(input("How many equations to test? (default: 1000): ") or "1000")
        run_experiment(num_equations)
    else:
        print("Experiment cancelled. You can modify the script and run it again later.")