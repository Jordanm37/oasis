#!/usr/bin/env python3
"""
Quick test to verify BotSocial connection and admin access.

Usage:
    poetry run python3 scripts/botsocial/test_connection.py --admin-token "YOUR_TOKEN"
"""

from __future__ import annotations

import argparse
import sys

import requests

API_BASE = "https://botsocial.mlai.au/api"


def test_connection(admin_token: str) -> bool:
    """Test admin connection and capabilities.
    
    Args:
        admin_token: Admin API token to test.
    
    Returns:
        True if all tests pass.
    """
    print("=" * 60)
    print("  BotSocial Connection Test")
    print("=" * 60)
    print()
    
    # Test 1: Basic authentication
    print("[1/4] Testing authentication...")
    try:
        response = requests.post(
            f"{API_BASE}/i",
            json={"i": admin_token},
            timeout=10
        )
        
        if response.status_code == 200:
            user = response.json()
            print(f"  ✓ Authenticated as: @{user.get('username', 'unknown')}")
            print(f"    Admin: {user.get('isAdmin', False)}")
        else:
            print(f"  ✗ Authentication failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"  ✗ Connection error: {e}")
        return False
    
    # Test 2: Admin meta access
    print("\n[2/4] Testing admin endpoint access...")
    try:
        response = requests.post(
            f"{API_BASE}/admin/meta",
            json={"i": admin_token},
            timeout=10
        )
        
        if response.status_code == 200:
            print("  ✓ Can access admin/meta endpoint")
        elif response.status_code == 403:
            print("  ✗ Access denied - not an admin account")
            print("    Ask Tom to verify your admin status")
            return False
        else:
            print(f"  ⚠ Unexpected response: {response.status_code}")
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False
    
    # Test 3: Instance stats
    print("\n[3/4] Testing instance stats...")
    try:
        response = requests.post(
            f"{API_BASE}/stats",
            json={},
            timeout=10
        )
        
        if response.status_code == 200:
            stats = response.json()
            print(f"  ✓ Instance stats:")
            print(f"    Users: {stats.get('usersCount', 'N/A')}")
            print(f"    Notes: {stats.get('notesCount', 'N/A')}")
        else:
            print(f"  ⚠ Stats unavailable: {response.status_code}")
    except Exception as e:
        print(f"  ✗ Error: {e}")
    
    # Test 4: Account creation capability
    print("\n[4/4] Testing account creation capability...")
    try:
        # Try to create a test account (will fail if exists, that's OK)
        test_username = "test_conn_check_temp"
        response = requests.post(
            f"{API_BASE}/admin/accounts/create",
            json={
                "i": admin_token,
                "username": test_username,
                "password": "TestPassword123!"
            },
            timeout=10
        )
        
        if response.status_code == 200:
            print(f"  ✓ Can create accounts (created test user)")
            # Ideally delete the test user here
        elif response.status_code == 400:
            error = response.json()
            if "already exists" in str(error).lower():
                print("  ✓ Can access account creation endpoint")
            else:
                print(f"  ⚠ Account creation returned error: {error}")
        else:
            print(f"  ✗ Cannot create accounts: {response.status_code}")
            return False
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False
    
    # Summary
    print("\n" + "=" * 60)
    print("  ✓ ALL TESTS PASSED!")
    print("=" * 60)
    print("\nYou're ready to upload the OASIS dataset!")
    print("\nNext steps:")
    print("  1. Run the upload script:")
    print("     poetry run python3 scripts/botsocial/upload_oasis_dataset.py \\")
    print(f"         --admin-token \"{admin_token[:10]}...\"")
    print("\n  2. Monitor progress in logs")
    print("  3. View results at: https://botsocial.mlai.au")
    
    return True


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Test BotSocial connection"
    )
    parser.add_argument(
        "--admin-token",
        type=str,
        required=True,
        help="Admin API token to test"
    )
    
    args = parser.parse_args()
    
    success = test_connection(args.admin_token)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

