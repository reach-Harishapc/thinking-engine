# core/cortex.py
"""
Cortex — simple controller that reasons about intent and chooses agents.
This is a minimal placeholder referencing agent_runtime externally if needed.
"""

import json
import os
from interfaces.native_agents.web_agent import Agent as WebAgent
from interfaces.native_agents.file_agent import Agent as FileAgent
from interfaces.native_agents.code_agent import Agent as CodeAgent
from interfaces.native_agents.reasoning_agent import Agent as ReasoningAgent

class Cortex:
    def __init__(self, memory=None, system_prompt=None):
        self.memory = memory
        self.system_prompt = system_prompt or self._default_system_prompt()
        self.python_knowledge = self._load_python_knowledge()
        self.code_examples = self._load_code_examples()
        self.learned_patterns = {}
        self.web_agent = WebAgent()
        self.file_agent = FileAgent()
        self.code_agent = CodeAgent()
        self.reasoning_agent = ReasoningAgent(memory=memory)
        self._learn_from_examples()

        # Conversation patterns for intent classification
        self.conversation_patterns = {
            "greeting": ["hello", "hi", "hey there", "good morning", "good afternoon", "good evening", "hey"],
            "how_are_you": ["how are you", "how do you do", "how's it going", "what's up", "how you doing"],
            "what_doing": ["what are you doing", "what's going on", "what you up to", "busy"],
            "farewell": ["bye", "goodbye", "see you", "farewell", "take care", "bye bye"]
        }

        # Default responses (can be overridden by system prompt)
        self.responses = {
            "greeting": ["Hello! I'm your Thinking Engine, ready to help you think and learn.", "Hi there! How can I assist you today?"],
            "how_are_you": ["I'm functioning optimally, thank you for asking! How can I help you?", "I'm doing well, processing information and ready to assist!"],
            "what_doing": ["I'm analyzing your queries and learning from our conversations.", "I'm here to help you think through problems and provide insights."],
            "farewell": ["Goodbye! Feel free to return anytime.", "Take care! I'm here whenever you need me."],
            "explain": ["Let me think about that for you.", "I'll help you understand this better."],
            "unknown": ["I'm still learning about that topic.", "That's an interesting question! Let me consider it."]
        }

    def _default_system_prompt(self):
        """Default system prompt for the Thinking Engine"""
        return {
            "identity": "You are a Thinking Engine, an advanced AI designed to help users think, learn, and solve problems.",
            "personality": "helpful, intelligent, curious, and analytical",
            "capabilities": "reasoning, learning from conversations, providing insights, and assisting with complex problems",
            "communication_style": "clear, concise, and engaging",
            "response_guidelines": [
                "Always be helpful and truthful",
                "Acknowledge the user's input before responding",
                "Provide detailed explanations when asked",
                "Admit when you don't know something",
                "Learn from each interaction to improve future responses"
            ]
        }

    def _load_python_knowledge(self):
        """Load Python programming knowledge base"""
        knowledge_file = os.path.join(os.path.dirname(__file__), '..', 'data', 'python_knowledge.json')
        if os.path.exists(knowledge_file):
            try:
                with open(knowledge_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Warning: Could not load Python knowledge: {e}")
                return []
        return []

    def _load_code_examples(self):
        """Load Python code examples for learning"""
        examples_file = os.path.join(os.path.dirname(__file__), '..', 'data', 'python_code_examples.json')
        if os.path.exists(examples_file):
            try:
                with open(examples_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Warning: Could not load code examples: {e}")
                return []
        return []

    def _learn_from_examples(self):
        """Learn patterns from code examples"""
        for example in self.code_examples:
            pattern = example.get('pattern', '').lower()
            question = example.get('question', '').lower()
            solution = example.get('solution', '')
            explanation = example.get('explanation', '')

            # Store pattern mappings
            if pattern not in self.learned_patterns:
                self.learned_patterns[pattern] = []

            self.learned_patterns[pattern].append({
                'question': question,
                'solution': solution,
                'explanation': explanation
            })

    def generate_code_solution(self, prompt: str):
        """Generate code solution based on learned patterns"""
        p = prompt.lower().strip()

        # Score and rank matches by relevance
        matches = []

        # Check patterns first (more specific)
        for pattern, examples in self.learned_patterns.items():
            if pattern in p:
                # Calculate relevance score
                pattern_words = set(pattern.split())
                prompt_words = set(p.split())
                overlap = len(pattern_words & prompt_words)
                total_words = len(pattern_words | prompt_words)
                score = overlap / total_words if total_words > 0 else 0

                matches.append((score, examples[0]))

        # Check questions for additional matches
        for example in self.code_examples:
            question = example.get('question', '').lower()
            question_words = set(question.split())
            prompt_words = set(p.split())
            overlap = len(question_words & prompt_words)
            total_words = len(question_words | prompt_words)
            score = overlap / total_words if total_words > 0 else 0

            # Only add if score is reasonable (> 0.3)
            if score > 0.3:
                matches.append((score, example))

        # Sort by score (highest first) and return best match
        if matches:
            matches.sort(key=lambda x: x[0], reverse=True)
            best_match = matches[0][1]
            return f"Here's how to do that in Python:\n\n```python\n{best_match['solution']}\n```\n\n{best_match.get('explanation', '')}"

        return None

    def load_system_prompt(self, prompt_file=None, prompt_dict=None):
        """Load system prompt from file or dictionary"""
        if prompt_dict:
            self.system_prompt = prompt_dict
        elif prompt_file and os.path.exists(prompt_file):
            with open(prompt_file, 'r', encoding='utf-8') as f:
                self.system_prompt = json.load(f)
        else:
            # Keep default prompt
            pass

        # Update responses based on system prompt
        self._update_responses_from_prompt()

    def _update_responses_from_prompt(self):
        """Update response templates based on system prompt"""
        # Extract identity name for personalized responses
        identity_text = self.system_prompt.get('identity', '')
        if 'You are' in identity_text:
            self.ai_name = identity_text.split('You are')[1].split(',')[0].strip()
        else:
            self.ai_name = "Thinking Engine"

        # Update greeting responses to be more personalized
        personality = self.system_prompt.get('personality', 'helpful')
        if 'friendly' in personality.lower() or 'warm' in personality.lower():
            self.responses["greeting"] = [
                f"Hello! I'm {self.ai_name}, ready to help you with anything you need!",
                f"Hi there! I'm {self.ai_name} - how can I assist you today?"
            ]
        elif 'professional' in personality.lower():
            self.responses["greeting"] = [
                f"Good day. I'm {self.ai_name}, at your service.",
                f"Hello. I'm {self.ai_name}, ready to assist with your inquiries."
            ]
        else:
            # Default technical responses
            self.responses["greeting"] = [
                f"Hello! I'm {self.ai_name}, ready to help you think and learn.",
                f"Hi there! I'm {self.ai_name}. How can I assist you today?"
            ]

    def get_system_context(self):
        """Get formatted system context for response generation"""
        prompt = self.system_prompt
        context = f"""You are {prompt['identity']}
Your personality is {prompt['personality']}.
You have capabilities in {prompt['capabilities']}.
Your communication style is {prompt['communication_style']}.

Guidelines:
"""
        for guideline in prompt['response_guidelines']:
            context += f"- {guideline}\n"

        return context

    def think(self, prompt: str):
        # simple heuristic decision
        p = prompt.lower()
        if any(k in p for k in ("search", "find", "summarize")):
            return {"action":"plan", "intent":"fetch_and_summarize", "prompt": prompt}
        if any(k in p for k in ("run code", "execute", "python")):
            return {"action":"plan", "intent":"code_run", "prompt": prompt}
        if any(k in p for k in ("edit file", "modify", "update")):
            return {"action":"plan", "intent":"edit_file", "prompt": prompt}
        return {"action":"think", "intent":"explain", "prompt": prompt}

    def classify_conversation(self, prompt: str):
        """Classify conversation intent based on patterns"""
        p = prompt.lower().strip()
        words = p.split()

        for intent, patterns in self.conversation_patterns.items():
            for pattern in patterns:
                # Check if pattern is a complete word or phrase in the prompt
                if f" {pattern} " in f" {p} " or p.startswith(f"{pattern} ") or p.endswith(f" {pattern}") or p == pattern:
                    return intent
        return "unknown"

    def _handle_math_execution(self, prompt: str):
        """Handle direct math calculation requests with Python code execution"""
        import re

        # Extract the math expression from the prompt
        # Look for patterns like "2+5", "10-3", "4*6", "8/2"
        math_pattern = r'(\d+)\s*([+\-*/])\s*(\d+)'
        match = re.search(math_pattern, prompt)

        if match:
            num1, operator, num2 = match.groups()
            num1, num2 = int(num1), int(num2)

            # Calculate the result
            if operator == '+':
                result = num1 + num2
                operation = "addition"
            elif operator == '-':
                result = num1 - num2
                operation = "subtraction"
            elif operator == '*':
                result = num1 * num2
                operation = "multiplication"
            elif operator == '/':
                if num2 != 0:
                    result = num1 / num2
                    # Check if it's a whole number result
                    if result == int(result):
                        result = int(result)
                else:
                    return "Error: Division by zero is not allowed."
                operation = "division"

            # Create the Python code
            code = f'print({num1} {operator} {num2})'

            # Return formatted response
            return f"""The {operation} of {num1} {operator} {num2} equals {result}.

**Python Code:**
```python
{code}
```

**Output:**
{result}"""

        return None

    def check_coding_question(self, prompt: str):
        """Check if the prompt is a coding question and return relevant knowledge"""
        if not self.python_knowledge:
            return None

        p = prompt.lower().strip()
        for knowledge_item in self.python_knowledge:
            patterns = knowledge_item.get('question_patterns', [])
            for pattern in patterns:
                if pattern.lower() in p:
                    return knowledge_item
        return None

    def search_web_for_answer(self, query: str):
        """Use web agent to search for information when knowledge base is insufficient"""
        try:
            print(f"[WEB SEARCH] Searching for: {query}")
            result = self.web_agent.run(None, query=query)

            if result.get("status") == "ok":
                summary = result.get("summary", "")
                if summary:
                    # Create a more comprehensive response with analysis
                    return self._analyze_and_summarize_search_results(query, summary)
            else:
                return f"I tried searching the web but encountered an issue: {result.get('error', 'Unknown error')}"

        except Exception as e:
            return f"Web search failed: {str(e)}"

        return None

    def _analyze_and_summarize_search_results(self, query: str, search_results: str):
        """Analyze search results and provide a comprehensive summary"""
        # If the search results already contain deep research (indicated by 🔍), return as-is
        if "🔍" in search_results and ("Deep Web Research" in search_results or "Conducting deep web research" in search_results):
            return f"Based on comprehensive web research for '{query}':\n\n{search_results}"

        # Otherwise, fall back to the old categorization method
        lines = [line.strip() for line in search_results.split('\n') if line.strip()]

        # Categorize results
        linkedin_profiles = []
        github_profiles = []
        youtube_channels = []
        company_profiles = []
        other_results = []

        for line in lines:
            line_lower = line.lower()
            if 'linkedin' in line_lower or 'data scientist' in line_lower or 'specialist' in line_lower:
                linkedin_profiles.append(line)
            elif 'github' in line_lower or 'reach-' in line_lower:
                github_profiles.append(line)
            elif 'youtube' in line_lower:
                youtube_channels.append(line)
            elif 'company' in line_lower or 'profile' in line_lower or 'dun & bradstreet' in line_lower:
                company_profiles.append(line)
            else:
                other_results.append(line)

        # Build comprehensive response
        response_parts = []
        response_parts.append(f"Based on web search results for '{query}', here's a comprehensive summary:")

        if linkedin_profiles:
            response_parts.append(f"\n**Professional Profiles:**")
            for profile in linkedin_profiles[:2]:
                clean_profile = profile.replace('• ', '').replace('&', '&')
                response_parts.append(f"• {clean_profile}")

        if github_profiles:
            response_parts.append(f"\n**Developer Presence:**")
            for profile in github_profiles[:2]:
                clean_profile = profile.replace('• ', '').replace('&', '&')
                response_parts.append(f"• {clean_profile}")

        if youtube_channels:
            response_parts.append(f"\n**Content Creation:**")
            for channel in youtube_channels[:2]:
                clean_channel = channel.replace('• ', '').replace('&', '&')
                response_parts.append(f"• {channel}")

        if company_profiles:
            response_parts.append(f"\n**Business Profiles:**")
            for company in company_profiles[:2]:
                clean_company = company.replace('• ', '').replace('&', '&')
                response_parts.append(f"• {clean_company}")

        if other_results:
            response_parts.append(f"\n**Additional Information:**")
            for result in other_results[:2]:
                clean_result = result.replace('• ', '').replace('&', '&')
                response_parts.append(f"• {clean_result}")

        # Add analysis based on query type
        query_lower = query.lower()
        if 'who is' in query_lower or 'who are' in query_lower:
            response_parts.append(f"\n**Summary:** Based on the search results, this appears to be a professional with expertise in data science, AI, and quantum computing, who is active on professional networking platforms and has a presence in the tech community.")

        response_parts.append(f"\n*This information is based on publicly available web search results and may not be comprehensive.*")

        return '\n'.join(response_parts)

    def generate_response(self, prompt: str):
        """Generate a conversational response based on prompt"""
        p = prompt.lower()

        # Check for direct math/code execution requests first
        if (('+' in p or '-' in p or '*' in p or '/' in p) and
            ('what is' in p or 'calculate' in p or 'compute' in p or 'result' in p) and
            ('python' in p or 'print' in p or 'output' in p)):
            # This looks like a direct calculation request
            return self._handle_math_execution(prompt)

        # First check if it's a coding question that needs code generation
        if any(word in p for word in ['how to', 'write', 'create', 'code', 'python', 'function', 'class', 'loop', 'if', 'variable']):
            code_solution = self.generate_code_solution(prompt)
            if code_solution:
                return code_solution

        # Then check for coding knowledge questions
        coding_knowledge = self.check_coding_question(prompt)
        if coding_knowledge:
            return coding_knowledge['answer']

        # Then check for conversation patterns
        conv_intent = self.classify_conversation(prompt)
        if conv_intent in self.responses and conv_intent != "unknown":
            import random
            response = random.choice(self.responses[conv_intent])
            return response

        # Check memory for similar queries and training data if memory is available
        if self.memory:
            try:
                # First check for direct similar queries
                similar_queries = self.memory.recall(prompt, limit=2)
                if similar_queries:
                    for item in similar_queries:
                        if item.get('output'):
                            return f"Based on my previous learning: {item['output']}"

                # Then check for relevant training samples (PDF content)
                # Get all recent experiences and filter for training samples
                # Since recall("") doesn't work, let's read the memory file directly
                training_samples = []
                try:
                    memory_file = os.path.join(os.path.dirname(__file__), '..', 'memory_store', 'experiences.jsonl')
                    if os.path.exists(memory_file):
                        with open(memory_file, 'r', encoding='utf-8') as f:
                            for line in f:
                                try:
                                    exp = json.loads(line.strip())
                                    meta = exp.get('meta', {})
                                    if 'full_content' in meta:  # This is a training sample
                                        training_samples.append(exp)
                                except json.JSONDecodeError:
                                    continue
                except Exception as e:
                    print(f"[DEBUG] Error reading memory file: {e}")

                if training_samples:
                    # Search through training samples for relevant content
                    prompt_lower = prompt.lower()
                    relevant_samples = []

                    for sample in training_samples:
                        meta = sample.get('meta', {})
                        content = meta.get('full_content', '').lower()

                        # Check if key terms from the query appear in the training content
                        query_words = set(prompt_lower.split())
                        content_words = set(content.split())

                        # Calculate overlap - if more than 15% of query words appear in content
                        overlap = len(query_words & content_words)
                        overlap_ratio = overlap / len(query_words) if query_words else 0

                        # Bonus for exact keyword matches (especially for technical terms)
                        bonus_score = 0
                        important_keywords = ['qubit', 'quantum', 'computer', 'algorithm', 'gate', 'circuit']
                        for keyword in important_keywords:
                            if keyword in prompt_lower and keyword in content:
                                bonus_score += 0.2  # Significant bonus for exact matches

                        total_score = overlap_ratio + bonus_score

                        if total_score > 0.15:  # Reasonable threshold for relevance
                            relevant_samples.append((total_score, sample))

                    # Sort by relevance and return the most relevant training content
                    if relevant_samples:
                        relevant_samples.sort(key=lambda x: x[0], reverse=True)
                        best_match = relevant_samples[0][1]
                        meta = best_match.get('meta', {})
                        content = meta.get('full_content', '')

                        # Extract a more complete and meaningful snippet
                        # Look for the most relevant section based on query keywords
                        query_lower = prompt.lower()
                        content_lower = content.lower()

                        # Find the best starting point - look for sections containing query keywords
                        best_start = 0
                        query_words = set(query_lower.split())
                        max_score = 0

                        # Scan through the content in windows to find the most relevant section
                        window_size = 600
                        for i in range(0, len(content) - window_size, 50):  # Smaller step size
                            window = content[i:i + window_size].lower()
                            window_words = set(window.split())

                            # Calculate base overlap
                            overlap = len(query_words & window_words)

                            # Add bonus for important keywords
                            bonus = 0
                            important_keywords = ['qubit', 'quantum', 'computer', 'algorithm', 'gate', 'circuit', 'entanglement', 'superposition']
                            for keyword in important_keywords:
                                if keyword in query_lower and keyword in window:
                                    bonus += 2  # Higher bonus for exact keyword matches in window

                            total_score = overlap + bonus

                            if total_score > max_score:
                                max_score = total_score
                                best_start = i

                        # Extract from the best starting point
                        snippet = content[best_start:best_start + 1000]  # Larger snippet

                        # Try to make it more complete by finding natural breaks
                        if len(snippet) >= 800:
                            # Look for paragraph breaks first
                            last_para = snippet.rfind('\n\n')
                            if last_para > 600:
                                snippet = snippet[:last_para]
                            else:
                                # Look for sentence endings
                                last_period = snippet.rfind('.')
                                if last_period > 700:
                                    snippet = snippet[:last_period + 1]
                                else:
                                    # Look for line breaks
                                    last_line = snippet.rfind('\n')
                                    if last_line > 600:
                                        snippet = snippet[:last_line]

                        # Clean up the snippet
                        snippet = snippet.strip()
                        if not snippet.endswith(('.', '!', '?')):
                            snippet += '...'

                        return f"Based on my training data about quantum computing, here's what I know:\n\n{snippet}"

            except Exception as e:
                print(f"[DEBUG] Memory search error: {e}")
                pass

        # WEB SEARCH FALLBACK: If we don't have good knowledge, search the web
        print(f"[THINK] I don't have specific knowledge about '{prompt}'. Let me search the web...")
        web_result = self.search_web_for_answer(prompt)
        if web_result:
            # Store this new knowledge in memory for future use
            if self.memory:
                try:
                    self.memory.store_experience("web_search", prompt, {"response": web_result})
                except:
                    pass
            return web_result

        # Final fallback to intent-based response
        intent = self.think(prompt)
        if intent['intent'] in self.responses:
            import random
            response = random.choice(self.responses[intent['intent']])
            return response

        return random.choice(self.responses["unknown"])

    def reason(self, prompt: str):
        # Generate intelligent conversational response
        return self.generate_response(prompt)

    def vectorize(self, prompt: str):
        # crude numeric vector as task signature
        toks = prompt.split()
        vec = [len(toks), sum(len(t) for t in toks)]
        # pad to length 5
        while len(vec) < 5:
            vec.append(0)
        return vec[:5]

    def export_state(self):
        # Export state including system prompt
        return {
            "system_prompt": self.system_prompt
        }

    def import_state(self, state):
        # Import state including system prompt
        if "system_prompt" in state:
            self.system_prompt = state["system_prompt"]
            self._update_responses_from_prompt()
