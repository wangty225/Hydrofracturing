// 基于准备好的dom，初始化echarts实例
var myChart = echarts.init(document.getElementById('chart'), 'white', {renderer: 'canvas'});
$(
    function () {
        fetchData(chart);
        /*  setInterval(fetchData, 2000); */
    }
);

function fetchData() {
    $.ajax({
        type: "post",
        url: "http://127.0.0.1:8000/SandPlugRiskEvaluation/chart",
        dataType: 'json',
        success: function (result) {
            myChart.setOption({
                series: [
                    {
                        data: result.data
                    }
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
            restore: {},
            saveAsImage: {}
        }
    },
    tooltip: {
        trigger: 'axis',
        position: function (pt) {
            return [pt[0], '100%'];
        }
    },
    legend: {
        // data: ["JHW00923油压", "排出流量", "砂浓度", "液添1", "液添4"]
        data: ["JHW00923油压"]
    },
    xAxis: {
        type: 'time',
        boundaryGap: false,
        splitLine: {
            show: false
        }
    },
    yAxis: {
        type: 'value',
        boundaryGap: [0, '100%'],
        splitLine: {
            show: false
        }
    },
    series: [
        {
            name: 'JHW00923油压',
            type: 'line',
            showSymbol: false,
            data: [[0, 0], [1, 1], [2, 2], [3, 3]]
        }
    ]
};

// 使用刚指定的配置项和数据显示图表。
myChart.setOption(option);