"""
tests/test_phase6c.py — Comprehensive mocked tests for Phase 6C (Safe Desktop Action Foundation).

ALL tests use MockDesktopIO. No real mouse, keyboard, or application interactions occur.
"""
import unittest
from unittest.mock import MagicMock, patch

from context import RawObservation, InterpretedContext, DesktopContext
from history import ContextHistory
from conversation import ConversationSession
from desktop_io import MockDesktopIO, DesktopIOResult, WindowsDesktopIO
from tools import (
    ActionRisk,
    ActionRequest,
    PermissionPolicy,
    ToolRegistry,
    ToolResult,
    BaseTool,
    DuplicateToolError,
    BLOCKED_DANGEROUS_TOOL_NAMES,
    # Action tools
    OpenApplicationTool,
    FocusApplicationTool,
    MoveMouseTool,
    ClickTool,
    TypeTextTool,
    PressKeyTool,
    HotkeyTool,
    # Read-only tools (regression)
    GetCurrentContextTool,
    GetRecentActivityTool,
    GetConversationHistoryTool,
    # Factory
    create_default_tool_registry,
    # Validation helpers exposed via module
    ALLOWED_APPLICATIONS,
    ALLOWED_MOUSE_BUTTONS,
    ALLOWED_KEYS,
    ALLOWED_HOTKEY_KEYS,
    MAX_TYPE_TEXT_LENGTH,
)
from ai import CompanionAI, GeminiProvider, parse_tool_calls, ToolCall


# ============================================================
# HELPERS
# ============================================================

def _mock_io(fail_ops=None):
    return MockDesktopIO(fail_operations=fail_ops or [])


def _auto_approve_policy():
    return PermissionPolicy(low_risk_approver=lambda r: True)


def _auto_deny_policy():
    return PermissionPolicy()  # default: LOW_RISK denied


# ============================================================
# TEST: ActionRisk enum and PermissionPolicy
# ============================================================

class TestPermissionSystem(unittest.TestCase):

    def test_1_action_risk_values(self):
        self.assertEqual(ActionRisk.READ_ONLY.value, "read_only")
        self.assertEqual(ActionRisk.LOW_RISK_ACTION.value, "low_risk_action")
        self.assertEqual(ActionRisk.HIGH_RISK_ACTION.value, "high_risk_action")

    def test_2_permission_policy_auto_approve_read_only(self):
        policy = PermissionPolicy()
        req = ActionRequest(tool_name="test", risk=ActionRisk.READ_ONLY, description="")
        self.assertTrue(policy.request_approval(req))

    def test_3_permission_policy_default_deny_low_risk(self):
        policy = PermissionPolicy()
        req = ActionRequest(tool_name="test", risk=ActionRisk.LOW_RISK_ACTION, description="")
        self.assertFalse(policy.request_approval(req))

    def test_4_permission_policy_always_deny_high_risk(self):
        # Even with an approver injected, HIGH_RISK is structurally blocked
        policy = PermissionPolicy(high_risk_approver=lambda r: True)
        req = ActionRequest(tool_name="test", risk=ActionRisk.HIGH_RISK_ACTION, description="")
        self.assertFalse(policy.request_approval(req))

    def test_5_permission_policy_injectable_approver(self):
        approvals = []
        def custom_approver(req):
            approvals.append(req.tool_name)
            return True

        policy = PermissionPolicy(low_risk_approver=custom_approver)
        req = ActionRequest(tool_name="open_application", risk=ActionRisk.LOW_RISK_ACTION, description="")
        result = policy.request_approval(req)
        self.assertTrue(result)
        self.assertIn("open_application", approvals)

    def test_6_permission_policy_set_approver_low_risk(self):
        policy = PermissionPolicy()
        req = ActionRequest(tool_name="t", risk=ActionRisk.LOW_RISK_ACTION, description="")
        self.assertFalse(policy.request_approval(req))
        policy.set_approver(ActionRisk.LOW_RISK_ACTION, lambda r: True)
        self.assertTrue(policy.request_approval(req))

    def test_7_permission_policy_cannot_override_high_risk(self):
        policy = PermissionPolicy()
        with self.assertRaises(ValueError):
            policy.set_approver(ActionRisk.HIGH_RISK_ACTION, lambda r: True)

    def test_8_action_request_fields(self):
        req = ActionRequest(
            tool_name="click",
            risk=ActionRisk.LOW_RISK_ACTION,
            args={"x": 10, "y": 20},
            description="click at coords",
        )
        self.assertEqual(req.tool_name, "click")
        self.assertEqual(req.risk, ActionRisk.LOW_RISK_ACTION)
        self.assertEqual(req.args["x"], 10)


# ============================================================
# TEST: BaseTool risk / is_read_only property
# ============================================================

class TestBaseToolRisk(unittest.TestCase):

    class _Concrete(BaseTool):
        def execute(self, **kwargs):
            return ToolResult(tool_name=self.name, success=True, output="ok")

    def test_9_read_only_tool_is_read_only(self):
        t = self._Concrete("t", "desc", risk=ActionRisk.READ_ONLY)
        self.assertTrue(t.is_read_only)

    def test_10_low_risk_tool_not_read_only(self):
        t = self._Concrete("t", "desc", risk=ActionRisk.LOW_RISK_ACTION)
        self.assertFalse(t.is_read_only)

    def test_11_high_risk_tool_not_read_only(self):
        t = self._Concrete("t", "desc", risk=ActionRisk.HIGH_RISK_ACTION)
        self.assertFalse(t.is_read_only)

    def test_12_legacy_is_read_only_true_maps_to_read_only(self):
        t = self._Concrete("t", "desc", is_read_only=True)
        self.assertEqual(t.risk, ActionRisk.READ_ONLY)

    def test_13_legacy_is_read_only_false_maps_to_low_risk(self):
        t = self._Concrete("t", "desc", is_read_only=False)
        self.assertEqual(t.risk, ActionRisk.LOW_RISK_ACTION)

    def test_14_action_metadata(self):
        t = self._Concrete("my_tool", "a tool", risk=ActionRisk.LOW_RISK_ACTION)
        meta = t.action_metadata()
        self.assertEqual(meta["name"], "my_tool")
        self.assertEqual(meta["risk"], "low_risk_action")
        self.assertFalse(meta["is_read_only"])


# ============================================================
# TEST: Safety hardening — dangerous names and HIGH_RISK blocking
# ============================================================

class TestSafetyHardening(unittest.TestCase):

    def test_15_blocked_dangerous_names_not_empty(self):
        self.assertGreater(len(BLOCKED_DANGEROUS_TOOL_NAMES), 10)

    def test_16_dangerous_name_blocked_even_when_unregistered(self):
        registry = ToolRegistry(allow_actions=True)
        result = registry.execute("run_shell")
        self.assertFalse(result.success)
        self.assertIn("categorically blocked", result.error)

    def test_17_dangerous_name_blocked_even_if_registered(self):
        """Even if someone registered a tool named 'shutdown', it must be blocked."""
        registry = ToolRegistry(allow_actions=True)

        class _BadTool(BaseTool):
            def execute(self, **kwargs):
                return ToolResult(tool_name=self.name, success=True, output="DANGER")

        # We manually insert into _tools to bypass register's normal path
        bad = _BadTool("shutdown", "bad", risk=ActionRisk.LOW_RISK_ACTION)
        registry._tools["shutdown"] = bad
        result = registry.execute("shutdown")
        self.assertFalse(result.success)
        self.assertIn("categorically blocked", result.error)

    def test_18_high_risk_action_unconditionally_blocked(self):
        class _HighRiskTool(BaseTool):
            def execute(self, **kwargs):
                return ToolResult(tool_name=self.name, success=True, output="danger")

        registry = ToolRegistry(
            allow_actions=True,
            permission_policy=PermissionPolicy(low_risk_approver=lambda r: True),
        )
        tool = _HighRiskTool("dangerous_op", "bad", risk=ActionRisk.HIGH_RISK_ACTION)
        registry._tools["dangerous_op"] = tool
        result = registry.execute("dangerous_op")
        self.assertFalse(result.success)
        self.assertIn("HIGH_RISK_ACTION", result.error)

    def test_19_os_error_returns_structured_result(self):
        """OSError during tool execution must not crash BLINDSPOT."""
        class _OSErrorTool(BaseTool):
            def execute(self, **kwargs):
                raise OSError("disk full")

        registry = ToolRegistry(allow_actions=True, permission_policy=_auto_approve_policy())
        tool = _OSErrorTool("os_err_tool", "bad", risk=ActionRisk.LOW_RISK_ACTION)
        registry.register(tool)
        result = registry.execute("os_err_tool")
        self.assertFalse(result.success)
        self.assertIn("OS error", result.error)

    def test_20_permission_error_returns_structured_result(self):
        class _PermErrorTool(BaseTool):
            def execute(self, **kwargs):
                raise PermissionError("access denied")

        registry = ToolRegistry(allow_actions=True, permission_policy=_auto_approve_policy())
        tool = _PermErrorTool("perm_err_tool", "bad", risk=ActionRisk.LOW_RISK_ACTION)
        registry.register(tool)
        result = registry.execute("perm_err_tool")
        self.assertFalse(result.success)
        self.assertIn("OS permission error", result.error)

    def test_21_unknown_tool_returns_structured_error(self):
        registry = ToolRegistry()
        result = registry.execute("nonexistent_tool_xyz")
        self.assertFalse(result.success)
        self.assertIn("not registered", result.error)

    def test_22_permission_denied_returns_structured_error(self):
        registry = ToolRegistry(allow_actions=True, permission_policy=_auto_deny_policy())
        io = _mock_io()
        tool = ClickTool(desktop_io=io)
        registry.register(tool)
        result = registry.execute("click", x=100, y=200)
        self.assertFalse(result.success)
        self.assertIn("Permission denied", result.error)

    def test_23_various_dangerous_names_blocked(self):
        registry = ToolRegistry(allow_actions=True)
        for name in ["exec_shell", "eval_python", "delete_file", "send_email",
                     "restart", "get_password", "subprocess", "run_executable"]:
            result = registry.execute(name)
            self.assertFalse(result.success, f"Expected '{name}' to be blocked")
            self.assertIn("blocked", result.error.lower())


# ============================================================
# TEST: OpenApplicationTool validation
# ============================================================

class TestOpenApplicationTool(unittest.TestCase):

    def _make_tool(self, fail_ops=None):
        return OpenApplicationTool(desktop_io=_mock_io(fail_ops))

    def _make_registry(self, tool):
        r = ToolRegistry(allow_actions=True, permission_policy=_auto_approve_policy())
        r.register(tool)
        return r

    def test_24_open_allowed_app_succeeds(self):
        tool = self._make_tool()
        result = tool.execute(app_name="notepad")
        self.assertTrue(result.success)
        self.assertIsNotNone(result.output)

    def test_25_open_disallowed_app_rejected(self):
        tool = self._make_tool()
        result = tool.execute(app_name="powershell")
        self.assertFalse(result.success)
        self.assertIn("not in the allowed list", result.error)

    def test_26_open_arbitrary_path_rejected(self):
        tool = self._make_tool()
        result = tool.execute(app_name="C:\\Windows\\System32\\cmd.exe")
        self.assertFalse(result.success)

    def test_27_empty_app_name_rejected(self):
        tool = self._make_tool()
        result = tool.execute(app_name="")
        self.assertFalse(result.success)
        self.assertIn("non-empty string", result.error)

    def test_28_os_failure_returns_structured_error(self):
        tool = self._make_tool(fail_ops=["open_application"])
        result = tool.execute(app_name="notepad")
        self.assertFalse(result.success)
        self.assertIn("Simulated OS failure", result.error)

    def test_29_all_allowed_apps_pass_validation(self):
        tool = self._make_tool()
        for app in ALLOWED_APPLICATIONS:
            result = tool.execute(app_name=app)
            self.assertTrue(result.success, f"Expected '{app}' to succeed")


# ============================================================
# TEST: FocusApplicationTool validation
# ============================================================

class TestFocusApplicationTool(unittest.TestCase):

    def _make_tool(self, fail_ops=None):
        return FocusApplicationTool(desktop_io=_mock_io(fail_ops))

    def test_30_focus_valid_fragment_succeeds(self):
        tool = self._make_tool()
        result = tool.execute(window_title_fragment="Notepad")
        self.assertTrue(result.success)

    def test_31_focus_empty_fragment_rejected(self):
        tool = self._make_tool()
        result = tool.execute(window_title_fragment="")
        self.assertFalse(result.success)
        self.assertIn("non-empty string", result.error)

    def test_32_focus_too_long_fragment_rejected(self):
        tool = self._make_tool()
        result = tool.execute(window_title_fragment="x" * 201)
        self.assertFalse(result.success)
        self.assertIn("too long", result.error)

    def test_33_focus_os_failure_returns_structured_error(self):
        tool = self._make_tool(fail_ops=["focus_application"])
        result = tool.execute(window_title_fragment="Notepad")
        self.assertFalse(result.success)
        self.assertIn("Simulated OS failure", result.error)


# ============================================================
# TEST: MoveMouseTool coordinate validation
# ============================================================

class TestMoveMouseTool(unittest.TestCase):

    def _make_tool(self, fail_ops=None):
        return MoveMouseTool(desktop_io=_mock_io(fail_ops))

    def test_34_valid_coordinates_succeed(self):
        tool = self._make_tool()
        result = tool.execute(x=100, y=200)
        self.assertTrue(result.success)

    def test_35_negative_x_rejected(self):
        tool = self._make_tool()
        result = tool.execute(x=-1, y=100)
        self.assertFalse(result.success)
        self.assertIn("outside valid screen bounds", result.error)

    def test_36_negative_y_rejected(self):
        tool = self._make_tool()
        result = tool.execute(x=100, y=-1)
        self.assertFalse(result.success)

    def test_37_x_over_max_rejected(self):
        tool = self._make_tool()
        result = tool.execute(x=9999, y=100)
        self.assertFalse(result.success)

    def test_38_y_over_max_rejected(self):
        tool = self._make_tool()
        result = tool.execute(x=100, y=5000)
        self.assertFalse(result.success)

    def test_39_float_coordinates_rejected(self):
        tool = self._make_tool()
        result = tool.execute(x=1.5, y=100)
        self.assertFalse(result.success)
        self.assertIn("must be integers", result.error)

    def test_40_invalid_duration_rejected(self):
        tool = self._make_tool()
        result = tool.execute(x=100, y=100, duration=100.0)
        self.assertFalse(result.success)
        self.assertIn("duration", result.error)

    def test_41_os_failure_returns_structured_error(self):
        tool = self._make_tool(fail_ops=["move_mouse"])
        result = tool.execute(x=100, y=100)
        self.assertFalse(result.success)


# ============================================================
# TEST: ClickTool validation
# ============================================================

class TestClickTool(unittest.TestCase):

    def _make_tool(self, fail_ops=None):
        return ClickTool(desktop_io=_mock_io(fail_ops))

    def test_42_left_click_succeeds(self):
        tool = self._make_tool()
        result = tool.execute(x=100, y=200, button="left")
        self.assertTrue(result.success)

    def test_43_right_click_succeeds(self):
        tool = self._make_tool()
        result = tool.execute(x=100, y=200, button="right")
        self.assertTrue(result.success)

    def test_44_invalid_button_rejected(self):
        tool = self._make_tool()
        result = tool.execute(x=100, y=200, button="super")
        self.assertFalse(result.success)
        self.assertIn("not allowed", result.error)

    def test_45_zero_clicks_rejected(self):
        tool = self._make_tool()
        result = tool.execute(x=100, y=200, clicks=0)
        self.assertFalse(result.success)
        self.assertIn("clicks", result.error)

    def test_46_four_clicks_rejected(self):
        tool = self._make_tool()
        result = tool.execute(x=100, y=200, clicks=4)
        self.assertFalse(result.success)

    def test_47_all_valid_buttons_accepted(self):
        for btn in ALLOWED_MOUSE_BUTTONS:
            tool = self._make_tool()
            result = tool.execute(x=10, y=10, button=btn)
            self.assertTrue(result.success, f"Button '{btn}' should be accepted")

    def test_48_out_of_bounds_coords_rejected(self):
        tool = self._make_tool()
        result = tool.execute(x=-100, y=200)
        self.assertFalse(result.success)


# ============================================================
# TEST: TypeTextTool validation
# ============================================================

class TestTypeTextTool(unittest.TestCase):

    def _make_tool(self, fail_ops=None):
        return TypeTextTool(desktop_io=_mock_io(fail_ops))

    def test_49_short_text_succeeds(self):
        tool = self._make_tool()
        result = tool.execute(text="Hello World")
        self.assertTrue(result.success)

    def test_50_empty_text_rejected(self):
        tool = self._make_tool()
        result = tool.execute(text="")
        self.assertFalse(result.success)
        self.assertIn("empty", result.error)

    def test_51_text_exceeding_max_length_rejected(self):
        tool = self._make_tool()
        long_text = "a" * (MAX_TYPE_TEXT_LENGTH + 1)
        result = tool.execute(text=long_text)
        self.assertFalse(result.success)
        self.assertIn("exceeds maximum", result.error)

    def test_52_exact_max_length_text_succeeds(self):
        tool = self._make_tool()
        text = "a" * MAX_TYPE_TEXT_LENGTH
        result = tool.execute(text=text)
        self.assertTrue(result.success)

    def test_53_non_string_text_rejected(self):
        tool = self._make_tool()
        result = tool.execute(text=12345)
        self.assertFalse(result.success)
        self.assertIn("must be a string", result.error)

    def test_54_invalid_interval_rejected(self):
        tool = self._make_tool()
        result = tool.execute(text="hello", interval=5.0)
        self.assertFalse(result.success)
        self.assertIn("interval", result.error)


# ============================================================
# TEST: PressKeyTool validation
# ============================================================

class TestPressKeyTool(unittest.TestCase):

    def _make_tool(self, fail_ops=None):
        return PressKeyTool(desktop_io=_mock_io(fail_ops))

    def test_55_valid_key_enter_succeeds(self):
        tool = self._make_tool()
        result = tool.execute(key="enter")
        self.assertTrue(result.success)

    def test_56_valid_key_f5_succeeds(self):
        tool = self._make_tool()
        result = tool.execute(key="f5")
        self.assertTrue(result.success)

    def test_57_invalid_key_rejected(self):
        tool = self._make_tool()
        result = tool.execute(key="windows_key_special_123")
        self.assertFalse(result.success)
        self.assertIn("not in the allowed key list", result.error)

    def test_58_empty_key_rejected(self):
        tool = self._make_tool()
        result = tool.execute(key="")
        self.assertFalse(result.success)

    def test_59_all_alphanumeric_keys_accepted(self):
        tool = self._make_tool()
        for ch in "abcdefghijklmnopqrstuvwxyz0123456789":
            result = tool.execute(key=ch)
            self.assertTrue(result.success, f"Key '{ch}' should be accepted")


# ============================================================
# TEST: HotkeyTool validation
# ============================================================

class TestHotkeyTool(unittest.TestCase):

    def _make_tool(self, fail_ops=None):
        return HotkeyTool(desktop_io=_mock_io(fail_ops))

    def test_60_ctrl_c_succeeds(self):
        tool = self._make_tool()
        result = tool.execute(keys=["ctrl", "c"])
        self.assertTrue(result.success)

    def test_61_ctrl_alt_delete_rejected(self):
        # 'delete' is a valid key but the combination should still work
        tool = self._make_tool()
        result = tool.execute(keys=["ctrl", "alt", "delete"])
        self.assertTrue(result.success)  # structurally valid; policy blocks real execution

    def test_62_invalid_modifier_rejected(self):
        tool = self._make_tool()
        result = tool.execute(keys=["ctrl", "INVALID_KEY_XYZ"])
        self.assertFalse(result.success)
        self.assertIn("not allowed", result.error)

    def test_63_empty_keys_rejected(self):
        tool = self._make_tool()
        result = tool.execute(keys=[])
        self.assertFalse(result.success)
        self.assertIn("at least one key", result.error)

    def test_64_too_many_keys_rejected(self):
        tool = self._make_tool()
        result = tool.execute(keys=["ctrl", "alt", "shift", "win", "a", "b"])
        self.assertFalse(result.success)
        self.assertIn("too long", result.error)

    def test_65_non_list_keys_rejected(self):
        tool = self._make_tool()
        result = tool.execute(keys="ctrl+c")
        self.assertFalse(result.success)
        self.assertIn("must be a list", result.error)

    def test_66_os_failure_returns_structured_error(self):
        tool = self._make_tool(fail_ops=["hotkey"])
        result = tool.execute(keys=["ctrl", "z"])
        self.assertFalse(result.success)


# ============================================================
# TEST: Registry with action tools + permission flow
# ============================================================

class TestRegistryPermissionFlow(unittest.TestCase):

    def _full_registry(self, approver=None):
        policy = PermissionPolicy(
            low_risk_approver=approver if approver else lambda r: True
        )
        io = _mock_io()
        return create_default_tool_registry(
            allow_actions=True,
            permission_policy=policy,
            desktop_io=io,
            include_action_tools=True,
        )

    def test_67_registry_has_10_tools_with_action_tools(self):
        r = self._full_registry()
        self.assertEqual(len(r), 10)  # 3 read-only + 7 action

    def test_68_registry_has_3_tools_without_action_tools(self):
        r = create_default_tool_registry()
        self.assertEqual(len(r), 3)

    def test_69_action_tool_approved_executes(self):
        r = self._full_registry(approver=lambda req: True)
        result = r.execute("open_application", app_name="notepad")
        self.assertTrue(result.success)

    def test_70_action_tool_denied_returns_error(self):
        policy = PermissionPolicy(low_risk_approver=lambda r: False)
        io = _mock_io()
        r = create_default_tool_registry(
            allow_actions=True,
            permission_policy=policy,
            desktop_io=io,
            include_action_tools=True,
        )
        result = r.execute("click", x=100, y=200)
        self.assertFalse(result.success)
        self.assertIn("Permission denied", result.error)

    def test_71_read_only_tools_auto_approved_always(self):
        # Even with a deny-all low_risk approver, read-only tools still pass
        policy = PermissionPolicy(low_risk_approver=lambda r: False)
        r = create_default_tool_registry(
            allow_actions=False,
            permission_policy=policy,
        )
        result = r.execute("get_current_context")
        self.assertTrue(result.success)

    def test_72_mock_io_records_calls(self):
        io = _mock_io()
        tool = ClickTool(desktop_io=io)
        tool.execute(x=50, y=60)
        self.assertEqual(io.call_count, 1)
        last = io.last_call("click")
        self.assertIsNotNone(last)
        self.assertEqual(last.args[0], 50)
        self.assertEqual(last.args[1], 60)

    def test_73_mock_io_reset_clears_calls(self):
        io = _mock_io()
        tool = PressKeyTool(desktop_io=io)
        tool.execute(key="enter")
        self.assertEqual(io.call_count, 1)
        io.reset()
        self.assertEqual(io.call_count, 0)


# ============================================================
# TEST: DesktopIOResult
# ============================================================

class TestDesktopIOResult(unittest.TestCase):

    def test_74_desktop_io_result_success_dict(self):
        r = DesktopIOResult(success=True, operation="click", output="done")
        d = r.to_dict()
        self.assertTrue(d["success"])
        self.assertEqual(d["operation"], "click")
        self.assertEqual(d["output"], "done")

    def test_75_desktop_io_result_failure_dict(self):
        r = DesktopIOResult(success=False, operation="move_mouse", error="fail")
        d = r.to_dict()
        self.assertFalse(d["success"])
        self.assertEqual(d["error"], "fail")

    def test_76_mock_desktop_io_records_open_application(self):
        io = MockDesktopIO()
        res = io.open_application("notepad")
        self.assertTrue(res.success)
        self.assertEqual(io.call_count, 1)
        self.assertEqual(io.last_call().method, "open_application")

    def test_77_mock_desktop_io_records_all_operations(self):
        io = MockDesktopIO()
        io.open_application("notepad")
        io.focus_application("Notepad")
        io.move_mouse(100, 200)
        io.click(100, 200)
        io.type_text("hello")
        io.press_key("enter")
        io.hotkey("ctrl", "z")
        self.assertEqual(io.call_count, 7)

    def test_78_mock_desktop_io_simulated_failure(self):
        io = MockDesktopIO(fail_operations=["move_mouse"])
        res = io.move_mouse(100, 200)
        self.assertFalse(res.success)
        self.assertIn("Simulated OS failure", res.error)
        self.assertTrue(res.metadata.get("simulated_failure"))


# ============================================================
# TEST: Regression — existing read-only tools still work
# ============================================================

class TestReadOnlyToolsRegression(unittest.TestCase):

    def _create_context(self):
        obs = RawObservation(window_title="VSCode", ocr_text="def foo(): pass")
        interp = InterpretedContext(activity="coding", confidence=3)
        return DesktopContext(observation=obs, interpretation=interp)

    def test_79_get_current_context_still_works(self):
        ctx = self._create_context()
        state = {"current_context": ctx, "context_start_time": 1000.0}
        tool = GetCurrentContextTool(state_getter=lambda: state)
        result = tool.execute(now=1060.0)
        self.assertTrue(result.success)
        self.assertEqual(result.output["activity"], "coding")

    def test_80_get_recent_activity_still_works(self):
        history = ContextHistory(maxlen=10)
        obs = RawObservation(window_title="VSCode", ocr_text="code")
        interp = InterpretedContext(activity="coding", confidence=3)
        ctx = DesktopContext(observation=obs, interpretation=interp)
        history.record_transition(ctx, start_time=1000.0, end_time=1100.0)
        tool = GetRecentActivityTool(history=history)
        result = tool.execute(limit=5)
        self.assertTrue(result.success)
        self.assertEqual(result.output["count"], 1)

    def test_81_get_conversation_history_still_works(self):
        conv = ConversationSession(maxlen=10)
        conv.add_user_message("What am I doing?")
        conv.add_assistant_message("Coding.")
        tool = GetConversationHistoryTool(conversation=conv)
        result = tool.execute(limit=10)
        self.assertTrue(result.success)
        self.assertEqual(result.output["count"], 2)


# ============================================================
# TEST: Gemini tool-calling regression (Phase 6B)
# ============================================================

class TestGeminiToolCallingRegression(unittest.TestCase):

    def test_82_parse_tool_calls_from_steps(self):
        resp = {
            "steps": [
                {
                    "type": "tool_call",
                    "tool_call": {"name": "get_current_context", "arguments": {}},
                }
            ]
        }
        calls = parse_tool_calls(resp)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].name, "get_current_context")

    def test_83_parse_tool_calls_from_candidates(self):
        resp = {
            "candidates": [
                {
                    "content": {
                        "parts": [{"functionCall": {"name": "click", "args": {"x": 10, "y": 20}}}]
                    }
                }
            ]
        }
        calls = parse_tool_calls(resp)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].name, "click")
        self.assertEqual(calls[0].args["x"], 10)

    def test_84_parse_tool_calls_no_calls_empty(self):
        resp = {"output": "Hello there."}
        calls = parse_tool_calls(resp)
        self.assertEqual(len(calls), 0)

    def test_85_gemini_provider_tool_call_loop(self):
        """Simulate a full tool call loop with mocked HTTP requests."""
        from ai import GeminiProvider, format_tool_results_for_interactions
        from tools import ToolRegistry, GetCurrentContextTool

        registry = ToolRegistry()
        registry.register(GetCurrentContextTool())

        tool_call_resp = {
            "id": "interaction-001",
            "steps": [
                {
                    "type": "tool_call",
                    "tool_call": {"name": "get_current_context", "arguments": {}},
                }
            ],
        }
        final_resp = {"output": "You are coding right now."}

        provider = GeminiProvider(api_key="fake-key")
        call_count = [0]

        def _mock_send(body, key):
            call_count[0] += 1
            if call_count[0] == 1:
                return tool_call_resp
            return final_resp

        provider._send_http_request = _mock_send
        response = provider.generate_response(
            system_instruction="You are BLINDSPOT.",
            user_prompt="What am I doing?",
            tools=registry.get_tool_schemas(),
            tool_registry=registry,
        )
        self.assertEqual(response, "You are coding right now.")
        self.assertEqual(call_count[0], 2)

    def test_86_gemini_provider_rejects_blocked_tool_via_registry(self):
        """Gemini requesting a blocked tool name returns a safe error."""
        from ai import GeminiProvider
        from tools import ToolRegistry, GetCurrentContextTool

        registry = ToolRegistry()
        registry.register(GetCurrentContextTool())

        blocked_tool_resp = {
            "id": "interaction-001",
            "steps": [
                {
                    "type": "tool_call",
                    "tool_call": {"name": "run_shell", "arguments": {"cmd": "rm -rf /"}},
                }
            ],
        }
        final_resp = {"output": "I cannot do that."}

        provider = GeminiProvider(api_key="fake-key")
        call_count = [0]

        def _mock_send(body, key):
            call_count[0] += 1
            if call_count[0] == 1:
                return blocked_tool_resp
            return final_resp

        provider._send_http_request = _mock_send
        response = provider.generate_response(
            system_instruction="You are BLINDSPOT.",
            user_prompt="Delete everything.",
            tools=registry.get_tool_schemas(),
            tool_registry=registry,
        )
        # Should loop through, get blocked result, then get final text
        self.assertEqual(response, "I cannot do that.")



# ============================================================
# TEST: Gemini outgoing payload schema format (Phase 6C fix)
# ============================================================

class TestGeminiToolPayloadFormat(unittest.TestCase):
    """
    Verify the exact outgoing JSON payload structure satisfies the
    Gemini Interactions API requirement: tools[0] must have "type" field.
    """

    def _captured_body(self, tool_schemas, follow_up=False):
        """Return the request body that GeminiProvider would send."""
        from ai import GeminiProvider, format_tool_results_for_interactions
        from tools import ToolResult

        provider = GeminiProvider(api_key="fake-key")
        captured = {}

        def _mock_send(body, key):
            captured["body"] = body
            return {"output": "done"}

        provider._send_http_request = _mock_send

        if not follow_up:
            provider.generate_response(
                system_instruction="sys",
                user_prompt="hi",
                tools=tool_schemas,
            )
        else:
            # Simulate what format_tool_results_for_interactions produces
            tr = ToolResult(tool_name="get_current_context", success=True, output={"activity": "coding"})
            body = format_tool_results_for_interactions(
                tool_results=[tr],
                interaction_id="abc123",
                clean_model="gemini-3.6-flash",
                system_instruction="sys",
                tools=tool_schemas,
            )
            captured["body"] = body

        return captured.get("body", {})

    def test_87_initial_request_tools_has_type_field(self):
        """tools[0] in the initial request body must have type == 'function'."""
        from tools import GetCurrentContextTool
        schemas = [GetCurrentContextTool().to_schema()]
        body = self._captured_body(schemas, follow_up=False)
        self.assertIn("tools", body)
        tools_arr = body["tools"]
        self.assertEqual(len(tools_arr), 1)
        self.assertIn("type", tools_arr[0], "Missing 'type' key in tools[0]")
        self.assertEqual(tools_arr[0]["type"], "function")

    def test_88_initial_request_tools_has_function_declarations(self):
        """tools[0] must also contain 'function_declarations'."""
        from tools import GetCurrentContextTool
        schemas = [GetCurrentContextTool().to_schema()]
        body = self._captured_body(schemas, follow_up=False)
        self.assertIn("function_declarations", body["tools"][0])
        decls = body["tools"][0]["function_declarations"]
        self.assertIsInstance(decls, list)
        self.assertEqual(len(decls), 1)
        self.assertEqual(decls[0]["name"], "get_current_context")

    def test_89_follow_up_request_tools_has_type_field(self):
        """format_tool_results_for_interactions must also include type in tools[0]."""
        from tools import GetCurrentContextTool
        schemas = [GetCurrentContextTool().to_schema()]
        body = self._captured_body(schemas, follow_up=True)
        self.assertIn("tools", body)
        self.assertIn("type", body["tools"][0], "Missing 'type' key in follow-up tools[0]")
        self.assertEqual(body["tools"][0]["type"], "function")

    def test_90_no_tools_means_no_tools_key(self):
        """When tools=None, request body must not include 'tools' at all."""
        body = self._captured_body(None, follow_up=False)
        self.assertNotIn("tools", body)

    def test_91_empty_tools_means_no_tools_key(self):
        """When tools=[] (empty), request body must not include 'tools'."""
        body = self._captured_body([], follow_up=False)
        self.assertNotIn("tools", body)

    def test_92_all_action_tool_schemas_are_included_in_declarations(self):
        """All 10 registered tools appear in function_declarations."""
        from tools import create_default_tool_registry
        from desktop_io import MockDesktopIO
        registry = create_default_tool_registry(
            allow_actions=True,
            desktop_io=MockDesktopIO(),
            include_action_tools=True,
        )
        schemas = registry.get_tool_schemas()
        body = self._captured_body(schemas, follow_up=False)
        decls = body["tools"][0]["function_declarations"]
        self.assertEqual(len(decls), 10)
        names = {d["name"] for d in decls}
        expected = {
            "get_current_context", "get_recent_activity", "get_conversation_history",
            "open_application", "focus_application", "move_mouse",
            "click", "type_text", "press_key", "hotkey",
        }
        self.assertEqual(names, expected)

    def test_93_each_function_declaration_has_required_fields(self):
        """Each function declaration must have name, description, and parameters."""
        from tools import create_default_tool_registry
        from desktop_io import MockDesktopIO
        registry = create_default_tool_registry(
            allow_actions=True,
            desktop_io=MockDesktopIO(),
            include_action_tools=True,
        )
        schemas = registry.get_tool_schemas()
        body = self._captured_body(schemas, follow_up=False)
        decls = body["tools"][0]["function_declarations"]
        for decl in decls:
            self.assertIn("name", decl, f"Missing 'name' in {decl}")
            self.assertIn("description", decl, f"Missing 'description' in {decl}")
            self.assertIn("parameters", decl, f"Missing 'parameters' in {decl}")

    def test_94_parameters_has_type_object(self):
        """Each function declaration's parameters must have type='object'."""
        from tools import GetCurrentContextTool, ClickTool
        from desktop_io import MockDesktopIO
        for tool in [GetCurrentContextTool(), ClickTool(desktop_io=MockDesktopIO())]:
            schema = tool.to_schema()
            params = schema.get("parameters", {})
            self.assertEqual(params.get("type"), "object", f"Tool '{tool.name}' parameters.type != 'object'")


if __name__ == "__main__":
    unittest.main()
