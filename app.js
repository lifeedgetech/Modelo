var btnSignin = document.querySelector("#signin");
var btnSignup = document.querySelector("#signup");

var body = document.querySelector("body");


btnSignin.addEventListener("click", function () {
   body.className = "sign-in-js"; 
});

btnSignup.addEventListener("click", function () {
    body.className = "sign-up-js";
})

/*document.addEventListener('DOMContentLoaded', () => {
    const signinButton = document.getElementById('signin');
    signinButton.addEventListener('click', () => {
        window.location.href = 'upload.html';
    });
});
*/