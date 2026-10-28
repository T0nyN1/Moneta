// 右侧大纲中被省略号截断的条目，悬停时用浏览器提示显示全文
document.addEventListener("mouseover", (event) => {
  const link = event.target.closest(".page-toc a.nav-link");
  if (link) {
    link.title = link.scrollWidth > link.clientWidth ? link.textContent.trim() : "";
  }
});
