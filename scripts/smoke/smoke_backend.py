#!/usr/bin/env python3
from .smoke_common import wait_for_health, get_json


def main():
    health = wait_for_health()
    print("Health OK:", health)
    num_pages = int(health.get("num_pages", 0))
    png_count = int(health.get("png_count", 0))
    if num_pages < 300:
        raise SystemExit(f"Expected at least 300 pages, got {num_pages}")
    if png_count and abs(num_pages - png_count) / max(num_pages, 1) > 0.02:
        raise SystemExit(f"Dataset ({num_pages}) and PNG count ({png_count}) differ by >2%")
    pages = get_json("/api/parse/boston/pages")
    if not pages.get("pages"):
        raise SystemExit("No pages returned")
    if len(pages["pages"]) < 300:
        raise SystemExit(f"Expected at least 300 pages from /boston/pages, got {len(pages['pages'])}")
    demo = get_json("/api/parse/boston/demo")
    print("Demo OK, default_page", demo.get("default_page"))
    sample_page = demo.get("default_page") or pages["pages"][0]
    recipe = get_json(f"/api/parse/boston/{sample_page}/recipe")
    assert "ingredients_lines" in recipe, "ingredients_lines missing"
    assert "instruction_lines" in recipe, "instruction_lines missing"
    print("Recipe schema OK for page", sample_page)


if __name__ == "__main__":
    main()
