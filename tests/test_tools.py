import unittest
import threading
from context import RawObservation, InterpretedContext, DesktopContext
from history import ContextHistory
from conversation import ConversationSession
from tools import (
    ToolResult,
    BaseTool,
    ToolRegistry,
    DuplicateToolError,
    GetCurrentContextTool,
    GetRecentActivityTool,
    GetConversationHistoryTool,
    create_default_tool_registry,
)
from ai import CompanionAI, LLMProvider


class MockActionTool(BaseTool):
    """Test tool that simulates a mutating action tool."""
    def __init__(self):
        super().__init__(
            name="mock_action",
            description="A mutating action tool for testing safety boundaries.",
            parameters={"type": "object", "properties": {}},
            is_read_only=False,
        )

    def execute(self, **kwargs) -> ToolResult:
        return ToolResult(tool_name=self.name, success=True, output="Action performed.")


class MockExplodingTool(BaseTool):
    """Test tool that raises an unexpected exception."""
    def __init__(self):
        super().__init__(
            name="mock_exploding",
            description="Raises an exception during execution.",
            is_read_only=True,
        )

    def execute(self, **kwargs) -> ToolResult:
        raise RuntimeError("Something exploded inside the tool.")


class TestToolArchitecture(unittest.TestCase):

    def _create_context(self, activity: str = "coding", window_title: str = "VSCode", ocr_text: str = "def foo(): pass") -> DesktopContext:
        obs = RawObservation(window_title=window_title, ocr_text=ocr_text)
        interp = InterpretedContext(activity=activity, confidence=3)
        return DesktopContext(observation=obs, interpretation=interp)

    def test_1_tool_result_creation_and_serialization(self):
        """Test ToolResult attributes and dictionary serialization."""
        res = ToolResult(
            tool_name="test_tool",
            success=True,
            output={"key": "val"},
            error=None,
            metadata={"source": "unit_test"},
            timestamp=1000.0,
        )
        self.assertEqual(res.tool_name, "test_tool")
        self.assertTrue(res.success)
        self.assertEqual(res.output, {"key": "val"})
        self.assertIsNone(res.error)

        d = res.to_dict()
        self.assertEqual(d["tool_name"], "test_tool")
        self.assertTrue(d["success"])
        self.assertEqual(d["output"], {"key": "val"})
        self.assertEqual(d["metadata"]["source"], "unit_test")
        self.assertEqual(d["timestamp"], 1000.0)

    def test_2_tool_schema(self):
        """Test BaseTool schema generation."""
        tool = GetCurrentContextTool()
        schema = tool.to_schema()

        self.assertEqual(schema["name"], "get_current_context")
        self.assertIn("description", schema)
        self.assertIn("parameters", schema)
        self.assertEqual(schema["parameters"]["type"], "object")

    def test_3_tool_registry_registration_and_retrieval(self):
        """Test registering and retrieving tools."""
        registry = ToolRegistry()
        self.assertEqual(len(registry), 0)

        tool = GetCurrentContextTool()
        registry.register(tool)

        self.assertEqual(len(registry), 1)
        self.assertEqual(registry.get("get_current_context"), tool)
        self.assertIsNone(registry.get("nonexistent"))
        self.assertIn(tool, registry.list_tools())

    def test_4_tool_registry_duplicate_prevention(self):
        """Test that registering duplicate tool names raises DuplicateToolError."""
        registry = ToolRegistry()
        registry.register(GetCurrentContextTool())

        with self.assertRaises(DuplicateToolError):
            registry.register(GetCurrentContextTool())

    def test_5_tool_registry_unknown_tool_error(self):
        """Test that executing an unknown tool returns a structured error result."""
        registry = ToolRegistry()
        result = registry.execute("unknown_tool")

        self.assertIsInstance(result, ToolResult)
        self.assertFalse(result.success)
        self.assertIn("not registered", result.error)

    def test_6_tool_registry_exception_handling(self):
        """Test that exceptions raised by tools are caught and returned as structured errors."""
        registry = ToolRegistry()
        registry.register(MockExplodingTool())

        result = registry.execute("mock_exploding")
        self.assertFalse(result.success)
        self.assertIn("Something exploded inside the tool", result.error)

    def test_7_safety_boundary_blocks_action_tools(self):
        """Test that action tools are rejected when allow_actions=False."""
        registry = ToolRegistry(allow_actions=False)
        registry.register(MockActionTool())

        result = registry.execute("mock_action")
        self.assertFalse(result.success)
        self.assertIn("safety boundary", result.error)
        self.assertIn("disabled", result.error)

    def test_8_safety_boundary_allows_when_enabled(self):
        """Test that action tools execute only when allow_actions=True."""
        registry = ToolRegistry(allow_actions=True)
        registry.register(MockActionTool())

        result = registry.execute("mock_action")
        self.assertTrue(result.success)
        self.assertEqual(result.output, "Action performed.")

    def test_9_get_current_context_tool_active(self):
        """Test GetCurrentContextTool with an active desktop context."""
        ctx = self._create_context("coding", "VSCode - main.py", "print('hello world')")
        state = {
            "current_context": ctx,
            "context_start_time": 1000.0,
        }
        tool = GetCurrentContextTool(state_getter=lambda: state)
        result = tool.execute(now=1045.0)

        self.assertTrue(result.success)
        self.assertTrue(result.output["has_active_context"])
        self.assertEqual(result.output["activity"], "coding")
        self.assertEqual(result.output["window_title"], "VSCode - main.py")
        self.assertEqual(result.output["duration_seconds"], 45.0)
        self.assertEqual(result.output["duration_formatted"], "45s")
        self.assertIn("print('hello world')", result.output["ocr_snippet"])

    def test_10_get_current_context_tool_empty(self):
        """Test GetCurrentContextTool when no active context exists."""
        state = {"current_context": None, "context_start_time": None}
        tool = GetCurrentContextTool(state_getter=lambda: state)
        result = tool.execute()

        self.assertTrue(result.success)
        self.assertFalse(result.output["has_active_context"])

    def test_11_get_recent_activity_tool(self):
        """Test GetRecentActivityTool retrieves closed episodes and sequence."""
        history = ContextHistory(maxlen=10)
        ctx1 = self._create_context("coding", "VSCode")
        ctx2 = self._create_context("reading in a browser", "Docs")

        history.record_transition(ctx1, start_time=1000.0, end_time=1100.0)
        history.record_transition(ctx2, start_time=1100.0, end_time=1150.0)

        tool = GetRecentActivityTool(history=history)
        result = tool.execute(limit=5)

        self.assertTrue(result.success)
        self.assertEqual(result.output["count"], 2)
        self.assertEqual(result.output["flow"], ["coding", "reading in a browser"])
        self.assertEqual(len(result.output["episodes"]), 2)
        self.assertEqual(result.output["episodes"][0]["activity"], "coding")
        self.assertEqual(result.output["episodes"][1]["activity"], "reading in a browser")

    def test_12_get_conversation_history_tool(self):
        """Test GetConversationHistoryTool retrieves dialogue turns."""
        conv = ConversationSession(maxlen=10)
        conv.add_user_message("What am I doing?")
        conv.add_assistant_message("You are coding.")

        tool = GetConversationHistoryTool(conversation=conv)
        result = tool.execute(limit=10)

        self.assertTrue(result.success)
        self.assertEqual(result.output["count"], 2)
        turns = result.output["turns"]
        self.assertEqual(turns[0]["role"], "user")
        self.assertEqual(turns[0]["content"], "What am I doing?")
        self.assertEqual(turns[1]["role"], "assistant")
        self.assertEqual(turns[1]["content"], "You are coding.")

    def test_13_create_default_tool_registry(self):
        """Test create_default_tool_registry factory registers all default tools."""
        state = {"current_context": None}
        conv = ConversationSession()
        hist = ContextHistory()

        registry = create_default_tool_registry(state=state, conversation=conv, history=hist)
        self.assertEqual(len(registry), 3)
        self.assertIsNotNone(registry.get("get_current_context"))
        self.assertIsNotNone(registry.get("get_recent_activity"))
        self.assertIsNotNone(registry.get("get_conversation_history"))
        self.assertFalse(registry.allow_actions)

    def test_14_companion_ai_tool_integration(self):
        """Test CompanionAI exposes tools and execute_tool helper."""
        registry = ToolRegistry()
        registry.register(GetCurrentContextTool())

        ai = CompanionAI(tool_registry=registry)
        self.assertEqual(len(ai.tools), 1)

        result = ai.execute_tool("get_current_context")
        self.assertTrue(result.success)
        self.assertFalse(result.output["has_active_context"])

        # Unknown tool through companion
        fail_res = ai.execute_tool("unknown_tool")
        self.assertFalse(fail_res.success)


if __name__ == "__main__":
    unittest.main()
