var VideoBirthTime = new Date("2017-08-06 11:39:06.000").getTime();
console.log(VideoBirthTime);
var myVideoPlayer;

var global_delay;

var timeMarker;
var boundRange;


$(document).ready(function() {
    boundRange = 180000;
    mouseDown = false;

    global_delay = 0;

    var count = 0;

    var sendEvent = false;

    videojs("video").ready(function() {
        myVideoPlayer = this;
        myVideoPlayer.on('timeupdate', function() {
            var absoluteTime = cameraToAbsoluteTime(this.currentTime());

            if (myVideoPlayer.paused()) {
                sendEvent = true;
            } else if (count > 5) {
                count = 0;
                sendEvent = true;
            }
            count = count + 1;

            if (sendEvent == true) {
                pubsubz.publish('timeMarker', 
                    {'timeMarker': absoluteTime,
                     'minTime': absoluteTime - boundRange/4,
                     'maxTime': absoluteTime + 3*boundRange/4
                    });
                sendEvent = false;
            }
        });
    });


    // ========================================================================
    // custom data
    TimeSeries.createUI('necklace.csv', 'prox', ['proximity'], {ymin:2000, ymax:4000}, 0,
        function(elem) {
            pubsubz.subscribe('timeMarker', TimeSeries.updateTime.bind(elem));
        });

    AnnotationPoint.createUI('labelchewing.json', VideoBirthTime - 5000, VideoBirthTime + 120*60000, 200,'annotation', 
         function(elem) {
             pubsubz.subscribe('timeMarker', AnnotationPoint.updateTime.bind(elem));
    });


    TimeSeries.createUI('necklace.csv', 'lf', ['leanForward'], {ymin:0, ymax:120}, 0,
        function(elem) {
            pubsubz.subscribe('timeMarker', TimeSeries.updateTime.bind(elem));
        });

    TimeSeries.createUI('necklace.csv', 'energy', ['energy'], {ymin:0, ymax:5}, 0,
        function(elem) {
            pubsubz.subscribe('timeMarker', TimeSeries.updateTime.bind(elem));
        });


    TimeSeries.createUI('necklace.csv', 'ambient', ['ambient'], {ymin:0, ymax:1500}, 0,
        function(elem) {
            pubsubz.subscribe('timeMarker', TimeSeries.updateTime.bind(elem));
        });

    $('#adjustsync').click(function(e){
        console.log($('#videodelay').val());
        if (isNaN(parseInt($('#videodelay').val()))) {
          alert("Enter an integer");
    }

    global_delay = parseInt($('#videodelay').val());

    var absoluteTime = cameraToAbsoluteTime(videojs('video').currentTime());
    pubsubz.publish('timeMarker', {
        'timeMarker': absoluteTime,
        'minTime': absoluteTime - boundRange / 4,
        'maxTime': absoluteTime + 3 * boundRange / 4
        });
    });

});

window.onbeforeunload = function() {
  return 'Are you sure you want to leave?';
};

// ========================================================================
function cameraToAbsoluteTime(videoTime) {  
    //TODO: add missing JSON

    return 1000*videoTime + VideoBirthTime;
}

function absoluteTimeToCamera(absoluteTime) {
    return (absoluteTime - VideoBirthTime)/1000;
}
