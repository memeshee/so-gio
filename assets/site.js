(function () {
  var toggle = document.getElementById('marks');
  if (!toggle) return;
  var apply = function () { document.body.classList.toggle('hide-marks', !toggle.checked); };
  toggle.addEventListener('change', apply);
  apply();
})();
