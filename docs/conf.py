"""Sphinx 配置，中英文文档共用。

每种语言一个源目录（``docs/zh``、以后的 ``docs/en``），构建时用 ``-c docs`` 指向本文件::

    sphinx-build -c docs docs/zh build/docs/zh
    sphinx-build -c docs -D language=en docs/en build/docs/en
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import moneta  # noqa: E402
from docutils import nodes  # noqa: E402
from sphinx.search.zh import SearchChinese  # noqa: E402

project = "Moneta"
author = "Yi Ni"
copyright = "2026, Yi Ni"
version = release = moneta.__version__
language = "zh_CN"

extensions = [
    "myst_parser",
    "sphinx_copybutton",
]
source_suffix = {".md": "markdown"}
root_doc = "index"
exclude_patterns = ["_build", ".DS_Store"]

# ---------------------------------------------------------------- MyST

myst_enable_extensions = [
    "dollarmath",  # $...$ 行内公式与 $$...$$ 块公式
]
myst_heading_anchors = 6  # 为各级标题生成 GitHub 风格锚点，文档中的 file.md#锚点 链接依赖它

# ---------------------------------------------------------------- 复制按钮

# 复制代码时去掉 >>> 与 ... 提示符，只保留输入行
copybutton_prompt_text = r">>> |\.\.\. "
copybutton_prompt_is_regexp = True
copybutton_only_copy_prompt_lines = True

# ---------------------------------------------------------------- HTML

html_theme = "pydata_sphinx_theme"
html_title = f"Moneta {release}"
html_logo = "_static/moneta-logo.svg"
html_favicon = html_logo  # 浏览器标签页图标与导航栏图标一致
html_theme_options = {
    "logo": {"text": html_title},  # 有图标时主题默认不显示标题文字，这里显式保留
    "github_url": "https://github.com/T0nyN1/Moneta",
    "use_edit_page_button": True,
    "navbar_align": "left",
    "show_toc_level": 2,
}
templates_path = ["_templates"]  # 覆盖主题组件（侧边栏折叠按钮的中文文字）
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_js_files = ["custom.js"]
html_context = {  # “编辑此页”按钮指向的 GitHub 源文件
    "github_user": "T0nyN1",
    "github_repo": "Moneta",
    "github_version": "main",
    "doc_path": "docs/zh",
}


# ---------------------------------------------------------------- 标题锚点


def _use_github_anchors(app, doctree):
    """把标题的 HTML id 换成 GitHub 风格锚点（如 ``#scale-传播规则``）。

    docutils 默认的 id 会丢掉中文（``#scale``、``#id4``），网址与 GitHub 上的锚点不一致，
    且插入新标题后编号会整体漂移。这里把 MyST 算出的锚点设为主 id，原 id 保留为别名，
    并同步 MyST 的锚点表，让站内链接也指向新 id。
    """
    meta = app.env.metadata[app.env.docname]
    slugs = meta.get("myst_slugs")
    if not slugs:
        return
    by_id = {node["ids"][0]: node for node in doctree.findall(nodes.section) if node["ids"]}
    for slug, (line, old_id, text) in list(slugs.items()):
        node = by_id.get(old_id)
        if node is None or slug in doctree.ids:
            continue
        node["ids"].insert(0, slug)
        doctree.ids[slug] = node
        slugs[slug] = (line, slug, text)


def _short_nav_title(app, doctree):
    """导航中只显示标题破折号前的名字（``FixedArray — 定点数张量引擎`` → ``FixedArray``）。

    只改 Sphinx 的短标题（用于侧边栏、面包屑、上一页/下一页），页面大标题与浏览器标签页仍是完整标题。
    """
    env, docname = app.env, app.env.docname
    title = env.titles.get(docname)
    if title is None or " — " not in title.astext():
        return
    short = title.astext().split(" — ", 1)[0]
    env.titles[docname] = nodes.title("", short)
    # 侧边栏取的是页面目录树里 h1 的引用文字，不是 env.titles，两处都要改
    first_ref = next(env.tocs[docname].findall(nodes.reference), None)
    if first_ref is not None:
        first_ref.children = [nodes.Text(short)]


# ---------------------------------------------------------------- 中文搜索

# 索引端由 jieba 把正文切成词（“守恒”“分配”），但浏览器端默认只按空格和标点切查询，
# 输入“守恒分配”会被当成一个词而搜不到。这里按索引中已有的词对查询做正向最大匹配。
_ZH_SPLITTER = r"""
var splitQuery = (query) => {
  const terms = (typeof Search !== "undefined" && Search._index && Search._index.terms) || {};
  const out = [];
  for (const word of query.split(/[^\p{Letter}\p{Number}_\p{Emoji_Presentation}]+/gu)) {
    for (const run of word.match(/\p{Script=Han}+|[^\p{Script=Han}]+/gu) || []) {
      if (!/\p{Script=Han}/u.test(run)) {
        out.push(run);
        continue;
      }
      for (let i = 0; i < run.length; ) {
        let n = Math.min(run.length - i, 8);
        while (n > 1 && !Object.hasOwn(terms, run.slice(i, i + n))) n--;
        if (n > 1 || Object.hasOwn(terms, run[i])) out.push(run.slice(i, i + n));  // 索引里没有的单字丢掉
        i += n;
      }
    }
  }
  return out;
};
"""


class _ChineseSearch(SearchChinese):
    js_splitter_code = _ZH_SPLITTER


def setup(app):
    # 必须早于 Sphinx 的目录收集器（优先级 500），右侧大纲与左侧导航才会用上新 id
    app.connect("doctree-read", _use_github_anchors, priority=400)
    # 必须晚于 Sphinx 的标题收集器（优先级 500），否则短标题会被它覆盖
    app.connect("doctree-read", _short_nav_title, priority=600)
    app.add_search_language(_ChineseSearch)
