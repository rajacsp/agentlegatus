"""Property-based tests for Provider Capability Enforcement.

Property 28: Provider Capability Enforcement
Validates: Requirement 24.4 — When a provider doesn't support a required
capability, the system SHALL raise a CapabilityNotSupportedError.
"""

from typing import Any, Dict, List, Optional

import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from agentlegatus.providers.base import (
    BaseProvider,
    CapabilityNotSupportedError,
    ProviderCapability,
)
from agentlegatus.tools.tool import Tool


# ---------------------------------------------------------------------------
# Test providers with specific capability subsets
# ---------------------------------------------------------------------------

class ToolOnlyProvider(BaseProvider):
    """Provider that only supports TOOL_CALLING."""

    def _get_capabilities(self) -> List[ProviderCapability]:
        return [ProviderCapability.TOOL_CALLING]

    async def create_agent(self, agent_config: Dict[str, Any]) -> Any:
        return {}

    async def execute_agent(self, agent: Any, input_data: Any,
                            state: Optional[Dict[str, Any]] = None) -> Any:
        return {}

    async def invoke_tool(self, tool: Tool, tool_input: Dict[str, Any],
                          context: Dict[str, Any]) -> Any:
        return {}

    def export_state(self) -> Dict[str, Any]:
        return {}

    def import_state(self, state: Dict[str, Any]) -> None:
        pass

    def to_portable_graph(self, workflow: Any) -> Any:
        return {}

    def from_portable_graph(self, graph: Any) -> Any:
        return {}


class StreamingOnlyProvider(BaseProvider):
    """Provider that only supports STREAMING."""

    def _get_capabilities(self) -> List[ProviderCapability]:
        return [ProviderCapability.STREAMING]

    async def create_agent(self, agent_config: Dict[str, Any]) -> Any:
        return {}

    async def execute_agent(self, agent: Any, input_data: Any,
                            state: Optional[Dict[str, Any]] = None) -> Any:
        return {}

    async def invoke_tool(self, tool: Tool, tool_input: Dict[str, Any],
                          context: Dict[str, Any]) -> Any:
        return {}

    def export_state(self) -> Dict[str, Any]:
        return {}

    def import_state(self, state: Dict[str, Any]) -> None:
        pass

    def to_portable_graph(self, workflow: Any) -> Any:
        return {}

    def from_portable_graph(self, graph: Any) -> Any:
        return {}


class NoCapabilityProvider(BaseProvider):
    """Provider that supports no capabilities at all."""

    def _get_capabilities(self) -> List[ProviderCapability]:
        return []

    async def create_agent(self, agent_config: Dict[str, Any]) -> Any:
        return {}

    async def execute_agent(self, agent: Any, input_data: Any,
                            state: Optional[Dict[str, Any]] = None) -> Any:
        return {}

    async def invoke_tool(self, tool: Tool, tool_input: Dict[str, Any],
                          context: Dict[str, Any]) -> Any:
        return {}

    def export_state(self) -> Dict[str, Any]:
        return {}

    def import_state(self, state: Dict[str, Any]) -> None:
        pass

    def to_portable_graph(self, workflow: Any) -> Any:
        return {}

    def from_portable_graph(self, graph: Any) -> Any:
        return {}


class AllCapabilitiesProvider(BaseProvider):
    """Provider that supports every capability."""

    def _get_capabilities(self) -> List[ProviderCapability]:
        return list(ProviderCapability)

    async def create_agent(self, agent_config: Dict[str, Any]) -> Any:
        return {}

    async def execute_agent(self, agent: Any, input_data: Any,
                            state: Optional[Dict[str, Any]] = None) -> Any:
        return {}

    async def invoke_tool(self, tool: Tool, tool_input: Dict[str, Any],
                          context: Dict[str, Any]) -> Any:
        return {}

    def export_state(self) -> Dict[str, Any]:
        return {}

    def import_state(self, state: Dict[str, Any]) -> None:
        pass

    def to_portable_graph(self, workflow: Any) -> Any:
        return {}

    def from_portable_graph(self, graph: Any) -> Any:
        return {}


# ---------------------------------------------------------------------------
# Hypothesis strategies
# ---------------------------------------------------------------------------

capability_strategy = st.sampled_from(list(ProviderCapability))

capability_subset_strategy = st.lists(
    capability_strategy,
    min_size=0,
    max_size=len(ProviderCapability),
    unique=True,
)

provider_class_strategy = st.sampled_from([
    ToolOnlyProvider,
    StreamingOnlyProvider,
    NoCapabilityProvider,
    AllCapabilitiesProvider,
])


# ---------------------------------------------------------------------------
# Property 28: Provider Capability Enforcement
# ---------------------------------------------------------------------------

@given(
    provider_class=provider_class_strategy,
    capability=capability_strategy,
)
@settings(max_examples=50, deadline=2000)
def test_property_28_unsupported_capability_raises(
    provider_class: type,
    capability: ProviderCapability,
):
    """
    Property 28: Provider Capability Enforcement

    For any provider and any capability NOT in its declared capabilities,
    require_capability() SHALL raise CapabilityNotSupportedError.

    Validates: Requirement 24.4
    """
    provider = provider_class({})

    if capability not in provider.capabilities:
        with pytest.raises(CapabilityNotSupportedError) as exc_info:
            provider.require_capability(capability)

        error = exc_info.value
        assert error.capability == capability
        assert error.provider_name == provider_class.__name__
        assert set(error.supported_capabilities) == set(provider.capabilities)
    else:
        # Should NOT raise when capability IS supported
        provider.require_capability(capability)


@given(
    provider_class=provider_class_strategy,
    capability=capability_strategy,
)
@settings(max_examples=50, deadline=2000)
def test_property_28_supports_capability_consistency(
    provider_class: type,
    capability: ProviderCapability,
):
    """
    Property 28 Extension: supports_capability is consistent with
    _get_capabilities.

    For any provider, supports_capability(c) returns True iff c is in
    the list returned by _get_capabilities().

    Validates: Requirements 24.1, 24.2
    """
    provider = provider_class({})
    declared = provider._get_capabilities()

    if capability in declared:
        assert provider.supports_capability(capability) is True
    else:
        assert provider.supports_capability(capability) is False


@given(capability=capability_strategy)
@settings(max_examples=20, deadline=2000)
def test_property_28_no_capability_provider_always_raises(
    capability: ProviderCapability,
):
    """
    Property 28 Extension: A provider with zero capabilities must raise
    CapabilityNotSupportedError for every capability.

    Validates: Requirement 24.4
    """
    provider = NoCapabilityProvider({})

    with pytest.raises(CapabilityNotSupportedError) as exc_info:
        provider.require_capability(capability)

    assert exc_info.value.capability == capability
    assert exc_info.value.supported_capabilities == []


@given(capability=capability_strategy)
@settings(max_examples=20, deadline=2000)
def test_property_28_all_capabilities_provider_never_raises(
    capability: ProviderCapability,
):
    """
    Property 28 Extension: A provider with all capabilities must never
    raise CapabilityNotSupportedError.

    Validates: Requirements 24.1, 24.2
    """
    provider = AllCapabilitiesProvider({})
    # Should succeed for every capability without raising
    provider.require_capability(capability)


@given(
    provider_class=provider_class_strategy,
    capabilities=st.lists(capability_strategy, min_size=1, max_size=5),
)
@settings(max_examples=30, deadline=2000)
def test_property_28_error_message_contains_useful_info(
    provider_class: type,
    capabilities: List[ProviderCapability],
):
    """
    Property 28 Extension: CapabilityNotSupportedError message includes
    the provider name, the missing capability, and the supported list.

    Validates: Requirement 24.4
    """
    provider = provider_class({})

    for cap in capabilities:
        if cap not in provider.capabilities:
            with pytest.raises(CapabilityNotSupportedError) as exc_info:
                provider.require_capability(cap)

            msg = str(exc_info.value)
            assert provider_class.__name__ in msg
            assert cap.value in msg
