window.parent.$(function () {//使用window.parent调用父级jquery
    var head = document.getElementsByTagName("head").item(0);

    var linkList = window.parent.document.getElementsByTagName("link");//获取父窗口link标签对象列表
    for (var i = 0; i < linkList.length; i++) {
        var _link = document.createElement("link");
        _link.rel = 'stylesheet'
        _link.type = 'text/css';
        _link.href = linkList[i].href;
        head.appendChild(_link);
    }

    var scriptList = window.parent.document.getElementsByTagName("script");//获取父窗口script标签对象列表
    for (var i = 0; i < scriptList.length; i++) {
        var _script = document.createElement("script");
        _script.type = 'text/javascript';
        _script.src = scriptList[i].src;
        head.appendChild(_script);
    }
});
