 $(document).ready(function(){
  $(window).scroll(function(){
      if ($(this).scrollTop() > 100) {
          $('.scrollUpButton').fadeIn();
      } else {
          $('.scrollUpButton').fadeOut();
      }
  });
  $('.scrollUpButton').click(function(){
      $("html, body").animate({ scrollTop: 0 }, 500);
      return false;
  });
 });
$(document).ready(function() {
    $('a.abstract').click(function() {
        $(this).parent().parent().find(".abstract.hidden").toggleClass('open');
    });
    $('a.bibtex').click(function() {
        $(this).parent().parent().find(".bibtex.hidden").toggleClass('open');
    });
    $('a.iframe').click(function() {
        $(this).parent().parent().parent().find(".iframe.hidden").toggleClass('open');
    });

    $(".post-tag").click(function() {
        var id = "#div-" + $(this).attr("id");
        $(this).toggleClass("clicked");
        $(id).toggle();
    });
});
