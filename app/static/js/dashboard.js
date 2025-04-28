/**
 * Twitter Analytics Dashboard JavaScript
 * Handles interactive elements and AJAX requests
 */

document.addEventListener('DOMContentLoaded', function() {
    // Character counter for tweet inputs
    const tweetInputs = document.querySelectorAll('[id^="tweet-content"]');
    tweetInputs.forEach(input => {
        if (input) {
            const maxLength = 280;
            const counterElement = document.getElementById('char-count');
            
            if (counterElement) {
                // Initial count
                counterElement.textContent = `${maxLength - input.value.length} characters remaining`;
                
                // Update on input
                input.addEventListener('input', function() {
                    const remaining = maxLength - this.value.length;
                    counterElement.textContent = `${remaining} characters remaining`;
                    
                    // Change color when getting close to limit
                    if (remaining < 20) {
                        counterElement.className = 'text-danger';
                    } else {
                        counterElement.className = 'text-muted';
                    }
                });
            }
        }
    });
    
    // Handle quick tweet form submission via AJAX
    const quickTweetForm = document.getElementById('quick-tweet-form');
    if (quickTweetForm) {
        quickTweetForm.addEventListener('submit', function(e) {
            e.preventDefault();
            
            const tweetContent = document.getElementById('tweet-content');
            const content = tweetContent.value.trim();
            
            if (!content) {
                return;
            }
            
            // Disable form during submission
            const submitBtn = quickTweetForm.querySelector('button[type="submit"]');
            const originalBtnText = submitBtn.innerHTML;
            submitBtn.disabled = true;
            submitBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Posting...';
            
            // Send AJAX request to post tweet
            fetch('/api/tweets', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-Requested-With': 'XMLHttpRequest'
                },
                body: JSON.stringify({ content: content })
            })
            .then(response => response.json())
            .then(data => {
                if (data.error) {
                    alert('Error: ' + data.error);
                } else {
                    // Clear the form
                    tweetContent.value = '';
                    
                    // Update the character counter
                    const counterElement = document.getElementById('char-count');
                    if (counterElement) {
                        counterElement.textContent = '280 characters remaining';
                        counterElement.className = 'text-muted';
                    }
                    
                    // Refresh page to show the new tweet
                    // In a more advanced implementation, we'd update the DOM without reload
                    window.location.reload();
                }
            })
            .catch(error => {
                console.error('Error:', error);
                alert('An error occurred while posting your tweet.');
            })
            .finally(() => {
                // Re-enable the form
                submitBtn.disabled = false;
                submitBtn.innerHTML = originalBtnText;
            });
        });
    }
    
    // Follow/Unfollow buttons with AJAX
    const followButtons = document.querySelectorAll('[data-action="follow"], [data-action="unfollow"]');
    followButtons.forEach(button => {
        button.addEventListener('click', function(e) {
            e.preventDefault();
            
            const action = this.dataset.action;
            const username = this.dataset.username;
            const url = action === 'follow' 
                ? `/follow/${username}` 
                : `/unfollow/${username}`;
            
            // Show loading state
            const originalText = this.innerHTML;
            this.innerHTML = '<i class="fas fa-spinner fa-spin"></i>';
            this.disabled = true;
            
            // Send AJAX request
            fetch(url, {
                headers: {
                    'X-Requested-With': 'XMLHttpRequest'
                }
            })
            .then(response => {
                if (!response.ok) {
                    throw new Error('Network response was not ok');
                }
                return response.json();
            })
            .then(data => {
                // Toggle button state
                if (action === 'follow') {
                    this.innerHTML = '<i class="fas fa-user-minus"></i> Unfollow';
                    this.dataset.action = 'unfollow';
                    this.className = this.className.replace('btn-primary', 'btn-outline-danger');
                } else {
                    this.innerHTML = '<i class="fas fa-user-plus"></i> Follow';
                    this.dataset.action = 'follow';
                    this.className = this.className.replace('btn-outline-danger', 'btn-primary');
                }
                
                // Update follower count if element exists
                const followerCountElement = document.getElementById('follower-count');
                if (followerCountElement && data.follower_count !== undefined) {
                    followerCountElement.textContent = data.follower_count;
                }
            })
            .catch(error => {
                console.error('Error:', error);
                this.innerHTML = originalText;
                alert('An error occurred. Please try again.');
            })
            .finally(() => {
                this.disabled = false;
            });
        });
    });
    
    // Handle hashtag clicks for analytics
    const hashtagLinks = document.querySelectorAll('a[href^="/hashtag/"]');
    hashtagLinks.forEach(link => {
        link.addEventListener('click', function(e) {
            // Track hashtag click for analytics (could be implemented with a backend endpoint)
            const hashtag = this.textContent.trim();
            console.log(`Hashtag clicked: ${hashtag}`);
            
            // Continue with normal navigation
            // No e.preventDefault() here as we want the link to work normally
        });
    });
    
    // Tooltips initialization (if using Bootstrap tooltips)
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
});