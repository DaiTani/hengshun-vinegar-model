// Vinegar Flavor Monitoring System - Charts Module

const ChartColors = {
    primary: '#2C3E50',
    accent: '#E74C3C',
    success: '#27AE60',
    info: '#3498DB',
    warning: '#F39C12',
    purple: '#9B59B6',
    teal: '#1ABC9C',
};

const ChartDefaults = {
    fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif',
    borderRadius: 4,
};

// Default chart options
Chart.defaults.font.family = ChartDefaults.fontFamily;
Chart.defaults.color = '#666';

function createRadarChart(canvasId, labels, data, options = {}) {
    const ctx = document.getElementById(canvasId);
    if (!ctx) return null;
    
    return new Chart(ctx, {
        type: 'radar',
        data: {
            labels: labels,
            datasets: [{
                data: data,
                backgroundColor: options.backgroundColor || 'rgba(231, 76, 60, 0.2)',
                borderColor: options.borderColor || '#E74C3C',
                borderWidth: 2,
                pointBackgroundColor: options.borderColor || '#E74C3C',
                pointRadius: 4,
                pointHoverRadius: 6,
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            scales: {
                r: {
                    beginAtZero: true,
                    max: options.max || 10,
                    ticks: {
                        stepSize: options.stepSize || 2,
                        font: { size: 11 }
                    },
                    grid: {
                        color: 'rgba(0, 0, 0, 0.05)'
                    }
                }
            },
            plugins: {
                legend: { display: false },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            return context.label + ': ' + context.parsed.y.toFixed(2);
                        }
                    }
                }
            },
            ...options.chartOptions
        }
    });
}

function createLineChart(canvasId, labels, datasets, options = {}) {
    const ctx = document.getElementById(canvasId);
    if (!ctx) return null;
    
    return new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: datasets.map((ds, index) => ({
                label: ds.label,
                data: ds.data,
                borderColor: ds.color || Object.values(ChartColors)[index % 7],
                backgroundColor: ds.backgroundColor || 'transparent',
                fill: ds.fill || false,
                tension: 0.4,
                pointRadius: 0,
                pointHoverRadius: 4,
                ...ds
            }))
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            interaction: {
                mode: 'index',
                intersect: false,
            },
            scales: {
                x: {
                    title: {
                        display: !!options.xTitle,
                        text: options.xTitle,
                        font: { size: 12 }
                    },
                    ticks: { font: { size: 10 } },
                    grid: { display: false }
                },
                y: {
                    title: {
                        display: !!options.yTitle,
                        text: options.yTitle,
                        font: { size: 12 }
                    },
                    ticks: { font: { size: 10 } },
                    grid: { color: 'rgba(0, 0, 0, 0.05)' }
                }
            },
            plugins: {
                legend: {
                    position: options.legendPosition || 'bottom',
                    labels: {
                        font: { size: 10 },
                        boxWidth: 12,
                        usePointStyle: true
                    }
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            const label = context.dataset.label || '';
                            const value = typeof context.parsed.y === 'number' 
                                ? context.parsed.y.toFixed(3) 
                                : context.parsed.y;
                            return label + ': ' + value;
                        }
                    }
                }
            },
            ...options.chartOptions
        }
    });
}

function createBarChart(canvasId, labels, data, options = {}) {
    const ctx = document.getElementById(canvasId);
    if (!ctx) return null;
    
    return new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: options.label || 'Value',
                data: data,
                backgroundColor: options.backgroundColor || '#27AE60',
                borderRadius: ChartDefaults.borderRadius,
                barThickness: options.barThickness || 20,
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            indexAxis: options.indexAxis || 'y',
            scales: {
                x: {
                    beginAtZero: true,
                    max: options.max || undefined,
                    ticks: { font: { size: 10 } },
                    grid: { color: 'rgba(0, 0, 0, 0.05)' }
                },
                y: {
                    ticks: { font: { size: 10 } },
                    grid: { display: false }
                }
            },
            plugins: {
                legend: { display: false },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            return context.parsed.x.toFixed(2);
                        }
                    }
                }
            },
            ...options.chartOptions
        }
    });
}

function destroyChart(chart) {
    if (chart && typeof chart.destroy === 'function') {
        chart.destroy();
    }
}

function updateChartData(chart, newLabels, newDatasets) {
    if (!chart) return;
    
    if (newLabels) {
        chart.data.labels = newLabels;
    }
    
    if (newDatasets) {
        newDatasets.forEach((ds, index) => {
            if (chart.data.datasets[index]) {
                chart.data.datasets[index].data = ds.data;
            }
        });
    }
    
    chart.update('none');
}
