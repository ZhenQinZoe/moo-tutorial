window.MathJax={
  tex:{inlineMath:[['\\(','\\)']],displayMath:[['\\[','\\]']],processEscapes:true,tags:'ams'},
  options:{skipHtmlTags:['script','noscript','style','textarea','pre','code'],ignoreHtmlClass:'mermaid|search-results'},
  chtml:{scale:1,matchFontHeight:false},
  startup:{ready(){MathJax.startup.defaultReady();MathJax.startup.promise.then(()=>{document.documentElement.dataset.mathReady='true'})}}
};
