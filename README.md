# Hydrofracturing

## 常用命令

* 运行项目★
  
  - 打开```Terminal```，输入：
  - ```bash
    python manage.py runserver
    ```
  - 浏览器访问 ```http://127.0.0.1:8000```

* 创建项目(已完成) 
  
  1. ```bash
     django-admin startproject Hydrofracturing
     ```
  2. ```bash
     python manage.py migrate
     ```

* 配置全局变量
  
  - 创建```globalSettings.py```
  
  - 在settings.py中的TEMPLATES中加入
    
    ```python
    # 配置读取全局配置的函数
    'Hydrofracturing.globalSettings.readSettingFile',
    ```
  
  - HTML中使用方法：
    ```<a class="navbar-brand" href="#">{{ PROJECTNAME }}</a>```

* 创建APP★（自定义APP）
  
  1. ```python
     python manage.py startapp APPName
     ```
  
  2. 注册APP：
     
     - 找到函数名：```APPName文件夹下的apps.py```里面的函数名：```AppnameConfig```
     
     - 在```APPName文件夹下的settings.py中的INSTALLED_APPS=[......]```种添加该函数名：
       
       ```python
       INSTALLED_APPS=[
           .......,
           'APPName.apps.AppnameConfig',
       ]
       ```
  
  3. 在Hydrofracturing/urls.py中增加一行
     ```path('APPName/', include('APPName.urls'))```
  
  4. 在```APPName/urls.py的urlpatterns```中增加url的映射关系
     
     > 例如：SandPlugRiskEvaluation/urls.py中的urlpatterns中有path(r'eval', eval.test, name='eval')，表示http://127.0.0.1:8080/SandPlugRiskEvaluation/eval对应与后台的eval.py中的test函数，那么该test函数一定有一个（request）参数。

## 项目目录结构

```
Folder PATH listing
Volume serial number is 5204-4AC7
C:.
├───Hydrofracturing **项目默认目录**
├───data **存放上传文件数据的文件夹**
├───index **新建的首页App**
├───SandPlugRiskEvaluation **新建的砂堵判别App**
│ └───core  **后台的核心代码**
├───static **存放前端静态文件如css，js**
│ ├───assets ** temp.html（模板页面，包含侧边栏，导航条等等）用到的静态文件目录存放位置**
│ ├───index  **index如果有静态文件引入，存放于此，如css，js**
│ ├───plugins  **存放第三方包，如bootstrap等
│ └───SandPlugRiskEvaluation **存放SandPlugRiskEvaluation静态文件如css，js**
├───static_root **发布前使用命令`python manage.py collectstatic`更新一次(不用管)**
│ ├───admin
│ ├───assets
│ ├───index
│ └───SandPlugRiskEvaluation
└───templates **模板文件，如html**
 ├───index
 └───SandPlugRiskEvaluation


read.md，markdown文件，项目介绍。
```

## 最终发布程序（开发过程无需发布，此处仅用于记录方法）

* ```bash
  pip install pyinstaller
  ```

* 进入django项目所在路径，运行```pyi-makespec -D manage.py```

* 修改.spec文件
  
  ```
  datas=[
        (r'C:\Users\Bingooo\PycharmProjects\Hydrofracturing\static_root',r'.\static_root'),
        (r'C:\Users\Bingooo\PycharmProjects\Hydrofracturing\templates', r'.\templates')
  ],
  hiddenimports=[
    'index',
    'SandPlugRiskEvaluation',
    'Hydrofracturing.globalSettings'
  ],
  ```
  
  ★ 参考：https://blog.csdn.net/weixin_44147584/article/details/117356092

* 直接运行以下语句 ```pyinstaller manage.spec```

* 运行报错：RuntimeError: Script runserver does not exist.[7484] Failed to execute script manage。解决：
  
  > ```manage.exe runserver --noreload```

* 打开http://127.0.0.1:8000/时报错：TemplateDoesNotExist。解决：
  
  > 1. 来看看settings文件下的STATIC_URL、STATIC_ROOT、STATICFILES_DIRS三个变量
  > 
  > ```
  > STATIC_URL = '/static/'
  > STATIC_ROOT = os.path.join(BASE_DIR, 'static_root')     # 新增
  > STATICFILES_DIRS = [
  >         os.path.join(BASE_DIR, 'static'),
  > ]
  > ```
  > 
  > 2. STATIC_ROOT 是在部署的时候才发挥作用，它聚合了各个static文件夹下的所有静态文件。
  >    执行下面这行代码时，django会把所有的static文件都复制到STATIC_ROOT文件夹下。
  >    ```python manage.py collectstatic```

> 3. 在项目的urls文件下新增：
> 
> ```
> from django.conf.urls import static
> urlpatterns += static.static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
> ```
> 
> 4. 最后需要将STATIC_ROOT中的静态文件打包到.exe中，配置前面提到的manage.spec文件，在datas（datas用于添加额外的文件）中加入元组： 
> 
> ```
> (r'D:\PycharmProjects\o12307\static_root',r'.\static_root')
> ```
> 
> ```
> datas=[(r'D:\PycharmProjects\o12307\static_root',r'.\static_root'),(r'D:\PycharmProjects\o12307\templates', r'.\templates')],
> ```

#### final: ```pyinstaller manage.spec```

## 运行程序

* 在manage同级目录下，新建文件：run.bat

* 右键-编辑，输入：
  
  ```
  cd manage
  manage.exe runserver 8000  --noreload
  pause
  ```

* 运行run.bat，保持窗口开启即为应用开启

* 浏览器打开网址```http://127.0.0.1:8000```

---

<br/>

---

# ★★★ 迭代开发 ★★★

## 0. 环境安装配置

## 1. 新建一个App ★（自定义APP）

pycharm导入项目，下方Terminal，进入项目根目录

```bash
cd .......\Hydrofracturing
```

1. ```python
   python manage.py startapp 【APPName】
   注意:【APPName】换成需要的名字即可
   ```

2. 注册APP：
   
   - 找到函数名：`【APPName】文件夹下的apps.py`里面的函数名：`【AppnameConfig】`
   
   - 在`Hydrofracturing文件夹下的settings.py中的INSTALLED_APPS=[......]`种添加该函数名：
     
     ```python
     INSTALLED_APPS=[
       .......,
       'APPName.apps.AppnameConfig',
     ]
     ```

3. 在Hydrofracturing/urls.py中增加一行 `path('【APPName】/', include('【APPName】.urls'))`

4. 在`【APPName】/urls.py的urlpatterns`中增加url的映射关系
   
   > 例如：SandPlugRiskEvaluation/urls.py中的urlpatterns中有path(r'evalurl', eval.test, name='eval')，表示[http://127.0.0.1:8080/SandPlugRiskEvaluation/evalurl 对应与后台的eval.py中的test函数，那么该test函数一定有一个（request）参数。

## 2. 新建App首页http://127.0.0.1:8000/APPName/myPage访问

> 1. 在【APPName】/urls.py中的urlpatterns中加入```path(r'myPage', view.【函数名，如fname】, name='myPageName')```,【myPage可以修改且可以为空】

> 2. 在【APPName】/views.py中编写函数
> 
> ```python
> def fname(request):
>     # # 若得到请求的参数部分，根据请求方式GET or POST
>     # a = request.GET.get('【paramName】')
>     # 或
>     # b = request.POST.get('【paramName】')
> 
> 
>     # 此处可以调用其他函数等得到结果
> 
> 
>     return render(request, 【静态文件myPage.html路径】, （可选，要传递的参数）)
>     return HttpResponse(【内容】)
>     return redirect('/pretty/list/')
>     ....
>     return json_response(【content】)
>     等等方式返回
> 
>     eg:方法①
>     return render(request, './APPName/myPage.html', {'title': title})
>     方法②
>     return render(request, './APPName/myPage.html', {'title': title})
> ```

> 3. 若需要引入前端的第三方包时，放入```static\plugins```文件夹。例如```bootstrap```的引入。
>    
>    <mark>方法①：</mark>extends方法，拼接成一个页面，可以直接使用相同的样式表等等。
>    
>    > 在Hydrofracturing\templates下手动创建一个APPName的文件夹，前端页面myPage.html放在这里。
>    
>    ```html
>    {% extends "temp.html" %}
>    {% load static %}
>    {% block content %}
>        {{ block.super }}
>    
>        【其他内容不变，在此处编辑前端内容即可。】
>    
>    {% endblock content %}
>    ```
> 
>    <mark>     方法②：</mark>iframe方法，temp页面里面套页面，两个页面互相独立，需要通过继承使用父页面样式。
> 
> templates/appname/下新建page.html文件
> 
> ```html
> <table>
>     <th>
>         <td>表头1</td>
>         <td>表头2</td>
>     </th>
>     <tr>
>         <td>123333</td>
>         <td>456666</td>
>     </tr>
> </table>
> <p>page2222!!!!!!!!</p>
> This is a demo !!!!!
> ```
> 
> **views.py**定义函数fun1，返回该页面
> 
> ```python
> def getPage2(request):
>     if request.method=='GET':
>         return render(request, './demo11/demo_page2.html')
>     return None
> ```
> 
> 定义**url.py**路径path1，指向函数fun1。此时path1，此处为【appname】/getPage2，便可以作为render的第三个参数source了
> 
> ```python
> urlpatterns = [
>     # path('', views.函数名),
>     path('', views.indexdemo),
>     path('getPage2', views.getPage2) ## 加入此处！！
> ]
> ```
> 
> **views.py**定义函数f2，在render的第三个参数source里即可以返回path1
> 
> ```python
> def indexdemo2(request):
>     return render(request, 
>                   './temp.html', 
>                   {'source': '/demo11/getPage2'}
>            )
> ```
> 
> 定义url.py路径path2，指向函数f2
> 
> ```python
> urlpatterns = [
>     # path('', views.函数名),
>     path('', views.indexdemo),
>     path('demo2', views.indexdemo2),  ## 加入此处
>     path('getPage2', views.getPage2)
> ]
> ```

> 4. 前端引入bootstrap.css的方法，注意href的写法：
> 
> ```python
> <link rel="stylesheet" href="{% static 'plugins/bootstrap/css/bootstrap.css' %}">
> ```

> 5. 自定义的css，js文件存放位置：```static\【APPName】\```文件夹里
> 
> 6. 引入自定义的格式文件如`SandPlugRiskEvaluation\myChart.js`
> 
> ```html
> <script type="text/javascript" src="{% static 'SandPlugRiskEvaluation/myCharts.js' %}"></script>
> ```

> 7. 上传文件的位置，data/【APPName】文件夹中

## 3. 开发过程：

- 创建APP，并注册APP到settings.py中

- 编辑请求的url映射到哪个**函数**

- 编辑**此函数**，返回指向一个前端页面

- 编辑页面如下：

- > - 编辑 html，css。
  > 
  > - 编辑数据请求相关：
  > 
  > - > 1. ①在html的标签元素中（如**a**的href属性，**button**的onclick()调用js方法，**type=submit**提交表单），②或js使用ajax方法，
  >   > 
  >   > 2. 请求后台函数接口，
  >   > 
  >   > 3. 处理返回结果。javascript或jquery操作前端标签的刷新显示
  > 
  > - 编辑后台函数接口，实现功能返回给前端。

> iframe继承父窗口引入的JS,CSS
> ```    <script type="text/javascript" src="{% static 'assets/js/inherit.js' %}"></script>```

test add
