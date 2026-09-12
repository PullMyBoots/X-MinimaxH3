import json
import re
import subprocess
import unittest
from html.parser import HTMLParser
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class ReleaseI18nTests(unittest.TestCase):
    def test_language_switch_and_asset_order(self):
        html = (ROOT / "static" / "index.html").read_text(encoding="utf-8")
        self.assertIn('id="languageSwitch"', html)
        self.assertIn('data-locale="zh-CN"', html)
        self.assertIn('data-locale="en"', html)
        self.assertLess(html.index("i18n.js"), html.index("app.js"))

    def test_runtime_translation_contract(self):
        source = (ROOT / "static" / "i18n.js").read_text(encoding="utf-8")
        self.assertIn("h3serve_locale", source)
        self.assertIn("new URLSearchParams", source)
        self.assertIn("new MutationObserver", source)
        self.assertIn("h3serve:locale-changed", source)
        self.assertIn(".job-title", source)
        self.assertIn("textarea, code, pre", source)
        self.assertIn("Checking model files and the CUDA runtime", source)
        self.assertIn("Model components ready:", source)
        self.assertIn("V24 unified Pareto scheduler", source)
        self.assertIn("Controls CPU weight residency", source)

    def test_english_translation_does_not_corrupt_chinese_words(self):
        script = r"""
const fs = require('fs');
const vm = require('vm');
global.window = {location:{search:'?lang=en'}, dispatchEvent:()=>{}};
global.localStorage = {getItem:()=>null, setItem:()=>{}};
global.document = {addEventListener:()=>{}, documentElement:{}, querySelectorAll:()=>[]};
global.Node = {ELEMENT_NODE:1, TEXT_NODE:3, DOCUMENT_NODE:9};
global.NodeFilter = {SHOW_ELEMENT:1, SHOW_TEXT:4};
global.MutationObserver = function() {};
global.CustomEvent = function() {};
vm.runInThisContext(fs.readFileSync(process.argv[1], 'utf8'));
const values = ['分辨率', '二次采样常用分辨率驻点', '一采 6 · 二采 2 · 总计 8 步', '5.0 秒 · 121帧'];
process.stdout.write(JSON.stringify(values.map(value => window.H3I18n.t(value))));
"""
        result = subprocess.run(
            ["node", "-e", script, str(ROOT / "static" / "i18n.js")],
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        translated = json.loads(result.stdout)
        self.assertEqual(translated[0], "Resolution")
        self.assertEqual(translated[1], "Final-pass resolution detents")
        self.assertEqual(translated[2], "First 6 · Final 2 · 8 total steps")
        self.assertEqual(translated[3], "5.0s · 121 frames")
        self.assertFalse(any(re.search(r"[\u3400-\u9fff]", value) for value in translated))

    def test_static_english_surface_has_no_untranslated_chinese(self):
        class SurfaceParser(HTMLParser):
            def __init__(self):
                super().__init__()
                self.skip_depth = 0
                self.values = []

            def handle_starttag(self, tag, attrs):
                if tag in {"textarea", "code", "pre"}:
                    self.skip_depth += 1
                if self.skip_depth:
                    return
                for name, value in attrs:
                    if name in {"placeholder", "title", "aria-label"} and value:
                        self.values.append(value.strip())

            def handle_endtag(self, tag):
                if tag in {"textarea", "code", "pre"} and self.skip_depth:
                    self.skip_depth -= 1

            def handle_data(self, data):
                value = data.strip()
                if not self.skip_depth and value:
                    self.values.append(value)

        parser = SurfaceParser()
        parser.feed((ROOT / "static" / "index.html").read_text(encoding="utf-8"))
        sources = [
            value for value in parser.values
            if value != "中文" and re.search(r"[\u3400-\u9fff]", value)
        ]
        script = r"""
const fs = require('fs');
const vm = require('vm');
global.window = {location:{search:'?lang=en'}, dispatchEvent:()=>{}};
global.localStorage = {getItem:()=>null, setItem:()=>{}};
global.document = {addEventListener:()=>{}, documentElement:{}, querySelectorAll:()=>[]};
global.Node = {ELEMENT_NODE:1, TEXT_NODE:3, DOCUMENT_NODE:9};
global.NodeFilter = {SHOW_ELEMENT:1, SHOW_TEXT:4};
global.MutationObserver = function() {};
global.CustomEvent = function() {};
vm.runInThisContext(fs.readFileSync(process.argv[1], 'utf8'));
const values = JSON.parse(fs.readFileSync(0, 'utf8'));
process.stdout.write(JSON.stringify(values.map(value => window.H3I18n.t(value))));
"""
        result = subprocess.run(
            ["node", "-e", script, str(ROOT / "static" / "i18n.js")],
            input=json.dumps(sources, ensure_ascii=False),
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        leftovers = [
            value for value in json.loads(result.stdout)
            if re.search(r"[\u3400-\u9fff]", value)
        ]
        self.assertEqual(leftovers, [])

    def test_release_scripts_parse(self):
        for script in ("i18n.js", "app.js"):
            result = subprocess.run(
                ["node", "--check", str(ROOT / "static" / script)],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
