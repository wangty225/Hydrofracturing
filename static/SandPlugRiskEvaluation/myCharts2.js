// 基于准备好的dom，初始化echarts实例
var myChart = echarts.init(document.getElementById('chart'), 'white', {renderer: 'canvas'});
$(
    function () {
        /* fetchData(chart);
           setInterval(fetchData, 2000); */
    }
);
/*
                function fetchData() {
                    $.ajax({
                        type: "post",
                        url: "http://127.0.0.1:8000/SandPlugRiskEvaluation/chart",
                        dataType: 'json',
                        success: function (result) {
                            run(result.data);
                        }
                    });
                }
                */
$.get(
    'http://127.0.0.1:8000/SandPlugRiskEvaluation/chart',
    function (_rawData) {
        run(_rawData);
    }
);

function run(_rawData) {
    // var countries = ['Australia', 'Canada', 'China', 'Cuba', 'Finland', 'France', 'Germany', 'Iceland', 'India', 'Japan', 'North Korea', 'South Korea', 'New Zealand', 'Norway', 'Poland', 'Russia', 'Turkey', 'United Kingdom', 'United States'];
    const countries = ["JHW00923油压", "排出流量", "砂浓度", "液添1", "液添4"]
    const datasetWithFilters = [];
    const seriesList = [];
    echarts.util.each(countries, function (country) {
        var datasetId = 'dataset_' + country;
        datasetWithFilters.push({
            id: datasetId,
            fromDatasetId: 'dataset_raw',
            transform: {
                type: 'filter',
                config: {
                    and: [
                        {dimension: 'Year', gte: 1950},
                        {dimension: 'Country', '=': country}
                    ]
                }
            }
        });
        seriesList.push({
            type: 'line',
            datasetId: datasetId,
            showSymbol: false,
            name: country,
            endLabel: {
                show: true,
                formatter: function (params) {
                    return country + ': ' + params.value[1];
                }
            },
            labelLayout: {
                moveOverlap: 'shiftY'
            },
            emphasis: {
                focus: 'series'
            },
            encode: {
                x: 'Year',
                y: 'Income',
                label: ['Country', 'Income'],
                itemName: 'Year',
                tooltip: ['Income']
            }
        });
    });

    option = {
        animationDuration: 10000,
        dataset: [
            {
                id: 'dataset_raw',
                source: _rawData
            },
            ...datasetWithFilters
        ],
        title: {
            text: 'Income of Germany and France since 1950'
        },
        tooltip: {
            order: 'valueDesc',
            trigger: 'axis'
        },
        xAxis: {
            type: 'category',
            nameLocation: 'middle'
        },
        yAxis: {
            name: 'Income'
        },
        grid: {
            right: 140
        },
        series: seriesList
    };

    // 使用刚指定的配置项和数据显示图表。
    myChart.setOption(option);
}

/*
    var dom = document.getElementById("container");
    var myChart = echarts.init(dom);
    var app = {};


    function loadEcharts(echartJson) {
        var option = JSON.parse(echartJson);
        if (option && typeof option === "object") {
            myChart.setOption(option, true);
        }
    }
    */