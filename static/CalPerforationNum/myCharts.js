// 基于准备好的dom，初始化echarts实例
var myChart = echarts.init(document.getElementById('chart'), 'white', {renderer: 'canvas'});
    $(
    function () {
        // fetchData();
        /*  setInterval(fetchData, 2000); */
    }
);

// function ch_(){
//     var url=document.getElementById("file_").value;
//     url=url.split("\\");//这里要将 \ 转义一下
//     console.log(url)
//     alert("文件名 "+url);
// }

function fetchData() {
    $.ajax({
        type: "post",
        url: "/CalPerforationNum/chart",
        data:{
        },
        dataType: 'json',
        success: function (result) {
            myChart.setOption({
                xAxis: {
                    data: result.data[0]
                },
                series: [
                    {
                        name: '排出流量',
                        data: result.data[2]
                    },
                    {
                        name: '砂浓度',
                        data: result.data[3]
                    },
                    {
                        name: '油压',
                        data: result.data[1],
                    },
                ]
            });
        }
    });
}


// 指定图表的配置项和数据
var option = {
    title: {
        text: '压力数据曲线'
    },
    toolbox: {
        feature: {
            dataZoom: {
                yAxisIndex: 'none'
            },
            saveAsImage: {}
        }
    },
    tooltip: {
        trigger: 'axis',
    },
    legend: {
        data: ["油压", "排出流量", "砂浓度"]
    },
    xAxis: {
        type: 'category',
        // data: [0, 1, 2, 3, 4, 5],

    },
    yAxis: [
        {type: 'value',},
        {type: 'value',},
        {type: 'value',},
    ],
    series: [
        {
            name: '排出流量',
            type: 'line',
            showSymbol: false,
            // emphasis: {
            //     focus: 'series'
            // },
            // data: [2, 3, 4, 5, 6, 7]
        },
        {
            yAxisIndex: 1,
            name: '砂浓度',
            type: 'line',
            showSymbol: false,
            // emphasis: {
            //     focus: 'series'
            // },
            // data: [3, 4, 5, 6, 7, 8]
        },
        {
            yAxisIndex: 2,
            name: '油压',
            type: 'line',
            showSymbol: false,
            lineStyle: {
                color: 'red'
            },
            // emphasis: {
            //     focus: 'series'
            // },
            // data: [1, 2, 3, 4, 5, 6]
        },
    ]
};

// 使用刚指定的配置项和数据显示图表。
myChart.setOption(option);
myChart.dispatchAction({
    type: 'dataZoom',
    start: 0,
    end: 100
});

function loadChart(fileName){
    if (fileName==null)
        alert("空文件名")
    $.ajax({
        type: "post",
        url: "/CalPerforationNum/chart?load="+fileName,
        data:{
        },
        dataType: 'json',
        success: function (result) {
            myChart.setOption({
                xAxis: {
                    data: result.data[0]
                },
                series: [
                    {
                        name: '排出流量',
                        data: result.data[2]
                    },
                    {
                        name: '砂浓度',
                        data: result.data[3]
                    },
                    {
                        name: '油压',
                        data: result.data[1],
                    },
                ]
            });
        }
    });
}