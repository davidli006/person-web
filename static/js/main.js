console.log("kakak")

function showTip(tip, type) {
    var $tip = $('#tip');
    $tip.stop(true).prop('class', 'alert alert-' + type).text(tip).css('margin-left', - $tip.outerWidth() / 2).fadeIn(500).delay(2000).fadeOut(500);
}
