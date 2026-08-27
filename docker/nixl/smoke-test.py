#!/usr/bin/env python3
"""
nixl smoke test for the GB10 runtime image.

Creates a NIXL agent, lists available backend plugins, and asserts that the
UCX, LIBFABRIC, POSIX, OBJ and MOONCAKE plugins built in this image are
loadable. Does not require a GPU or a peer connection; it only checks that the
libraries/plugins are present and the Python API can initialize.
"""

import nixl

EXPECTED_BACKENDS = {"UCX", "LIBFABRIC", "POSIX", "OBJ", "MOONCAKE"}


def main() -> None:
    agent = nixl.nixl_agent("smoke_test_agent")
    available = set(agent.plugin_list)

    missing = EXPECTED_BACKENDS - available
    if missing:
        raise AssertionError(
            f"Missing expected nixl backend(s): {sorted(missing)}; "
            f"available plugins: {sorted(available)}"
        )

    present = EXPECTED_BACKENDS & available
    print("nixl smoke test OK:", sorted(present))


if __name__ == "__main__":
    main()