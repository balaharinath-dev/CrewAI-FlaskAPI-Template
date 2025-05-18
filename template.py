import os
from typing import Dict, List, Optional, Any, Type
from flask import Flask, request, jsonify
from crewai import Agent, Task, Crew, Process
from crewai.tools import BaseTool
from langchain_google_genai import GoogleGenerativeAI
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import datetime
import traceback

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)

# ----- Tool Implementation -----
class AnalysisToolInput(BaseModel):
    query: str = Field(description="Input query for analysis")
    depth: str = Field(description="Analysis depth level", default="standard")

class ContentAnalysisTool(BaseTool):
    name: str = "content_analyzer"
    description: str = "Analyzes content and extracts insights"
    args_schema: Type[BaseModel] = AnalysisToolInput

    def _run(self, query: str, depth: str = "standard") -> Dict[str, Any]:
        try:
            # Your analysis logic here
            return {
                "analysis": f"Deep analysis of '{query}'",
                "insights": ["insight1", "insight2"],
                "depth_level": depth
            }
        except Exception as e:
            return {"error": str(e)}

# ----- CrewAI Implementation -----
class ContentAnalysisCrew:
    def __init__(self):
        self.api_key = os.getenv("GEMINI_API_KEY")
        
        self.llm = GoogleGenerativeAI(
            model="gemini/gemini-1.5-flash",
            google_api_key=self.api_key,
            temperature=0.5,
        )
        
        self.analysis_tool = ContentAnalysisTool()
        self._setup_agents()
        self._setup_crew()
    
    def _setup_agents(self):
        """Initialize specialized agents"""
        self.research_agent = Agent(
            role="Research Analyst",
            goal="Gather and analyze content data",
            backstory="Expert in content research with 10 years experience",
            verbose=True,
            allow_delegation=False,
            llm=self.llm,
            tools=[self.analysis_tool]
        )
        
        self.strategy_agent = Agent(
            role="Strategy Consultant",
            goal="Develop actionable recommendations",
            backstory="Former marketing director turned AI strategist",
            verbose=True,
            allow_delegation=False,
            llm=self.llm
        )
    
    def _setup_crew(self):
        """Configure the crew workflow"""
        self.crew = Crew(
            agents=[self.research_agent, self.strategy_agent],
            tasks=[],
            verbose=True,
            process=Process.sequential
        )
    
    def _create_tasks(self, user_input: str) -> List[Task]:
        """Generate tasks based on user input"""
        research_task = Task(
            description=f"Analyze this content: {user_input}",
            expected_output="Comprehensive content analysis report",
            agent=self.research_agent,
            tools=[self.analysis_tool]
        )
        
        strategy_task = Task(
            description="Create marketing strategy based on analysis",
            expected_output="Actionable marketing recommendations",
            agent=self.strategy_agent,
            context=[research_task]
        )
        
        return [research_task, strategy_task]
    
    def analyze(self, user_input: str) -> Dict[str, Any]:
        """Execute the full analysis workflow"""
        tasks = self._create_tasks(user_input)
        self.crew.tasks = tasks
        return self.crew.kickoff()

# ----- Flask Routes -----
@app.route('/analyze', methods=['POST'])
def analyze_content():
    try:
        # Validate request
        if not request.is_json:
            return jsonify({"error": "Request must be JSON"}), 400
            
        data = request.get_json()
        user_input = data.get('query')
        
        if not user_input:
            return jsonify({"error": "Query parameter is required"}), 400
        
        # Process with CrewAI
        analyzer = ContentAnalysisCrew()
        result = analyzer.analyze(user_input)
        
        # Format response
        return jsonify({
            "status": "success",
            "data": {
                "analysis": result,
                "metadata": {
                    "model": "gemini-1.5-flash",
                    "timestamp": datetime.now().isoformat()
                }
            }
        })
        
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e),
            "traceback": traceback.format_exc()
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)