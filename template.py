import os
from typing import Dict, List, Optional, Any, Type
from flask import Flask, request, jsonify
from crewai import Agent, Task, Crew, Process
from crewai.tools import BaseTool
from langchain_google_genai import GoogleGenerativeAI
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)

# --------------------------
# 1. TOOL IMPLEMENTATION (EXACTLY AS PROVIDED)
# --------------------------
class MyToolInput(BaseModel):
    query: str = Field(description="Description of the query parameter")
    param2: str = Field(description="Description of second parameter")

class MyCustomTool(BaseTool):
    name: str = "my_custom_tool"
    description: str = "Description of what this tool does"
    args_schema: Type[BaseModel] = MyToolInput

    def _run(self, query: str, param2: str) -> Dict[str, Any]:
        """Implementation of the tool's functionality"""
        try:
            # Your tool logic here
            return {"result": f"Processed {query} with {param2}"}
        except Exception as e:
            return {"error": str(e)}

# --------------------------
# 2. CREWAI IMPLEMENTATION (EXACTLY AS PROVIDED)
# --------------------------
class MyCrew:
    def __init__(self):
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        
        # Initialize LLM
        self.llm = GoogleGenerativeAI(
            model="gemini/gemini-1.5-flash",
            google_api_key=self.gemini_api_key,
            temperature=0.5,
        )
        
        # Initialize tools
        self.custom_tool = MyCustomTool()
        
        # Setup agents and crew
        self._setup_agents()
        self._setup_crew()
    
    def _setup_agents(self):
        """Initialize all agents with their roles and tools"""
        self.primary_agent = Agent(
            role="Primary Agent Role",
            goal="What this agent aims to accomplish",
            backstory="Background of this agent",
            verbose=True,
            allow_delegation=False,
            llm=self.llm,
            tools=[self.custom_tool]
        )
        
        self.secondary_agent = Agent(
            role="Secondary Agent Role",
            goal="What this agent aims to accomplish",
            backstory="Background of this agent",
            verbose=True,
            allow_delegation=True,
            llm=self.llm
        )
    
    def _setup_crew(self):
        """Configure the crew with agents and process"""
        self.crew = Crew(
            agents=[self.primary_agent, self.secondary_agent],
            tasks=[],
            verbose=True,
            process=Process.sequential
        )
    
    def _create_tasks(self, user_input: str) -> List[Task]:
        """Create tasks based on user input"""
        task1 = Task(
            description=f"First task description based on: {user_input}",
            expected_output="What this task should produce",
            agent=self.primary_agent,
            tools=[self.custom_tool]
        )
        
        task2 = Task(
            description="Second task description",
            expected_output="What this task should produce",
            agent=self.secondary_agent,
            context=[task1]
        )
        
        return [task1, task2]
    
    def execute(self, user_input: str) -> Dict[str, Any]:
        """Execute the crew's workflow"""
        tasks = self._create_tasks(user_input)
        self.crew.tasks = tasks
        result = self.crew.kickoff()
        return {"result": result}

# --------------------------
# 3. FLASK ROUTES (NEW ADDITION)
# --------------------------
@app.route('/process', methods=['POST'])
def process():
    """Endpoint that executes the crew workflow"""
    try:
        data = request.get_json()
        
        if not data or 'input' not in data:
            return jsonify({"error": "Missing input parameter"}), 400
        
        crew = MyCrew()
        output = crew.execute(data['input'])
        
        return jsonify({
            "status": "success",
            "data": output
        })
    
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

# --------------------------
# 4. MAIN EXECUTION
# --------------------------
if __name__ == '__main__':
    # Run both the crew example and Flask app
    print("Running crew example...")
    crew = MyCrew()
    output = crew.execute("Example user input")
    print("Crew output:", output)
    
    print("\nStarting Flask server...")
    app.run(host='0.0.0.0', port=5000, debug=True)
