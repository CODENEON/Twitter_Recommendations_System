from flask import Blueprint, render_template, redirect, url_for, flash, request
from flask_login import login_required, current_user
from app import db
from app.models import User, Tweet
from flask_wtf import FlaskForm
from wtforms import StringField, TextAreaField, SubmitField
from wtforms.validators import DataRequired, Length, Email, ValidationError
from app.services.sentiment import get_user_sentiment_stats

user = Blueprint('user', __name__)

class ProfileForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired(), Length(min=2, max=64)])
    email = StringField('Email', validators=[DataRequired(), Email()])
    bio = TextAreaField('Bio', validators=[Length(max=160)])
    submit = SubmitField('Update Profile')
    
    def __init__(self, original_username, original_email, *args, **kwargs):
        super(ProfileForm, self).__init__(*args, **kwargs)
        self.original_username = original_username
        self.original_email = original_email
    
    def validate_username(self, username):
        if username.data != self.original_username:
            user = User.query.filter_by(username=username.data).first()
            if user is not None:
                raise ValidationError('Username already taken. Please use a different username.')
    
    def validate_email(self, email):
        if email.data != self.original_email:
            user = User.query.filter_by(email=email.data).first()
            if user is not None:
                raise ValidationError('Email already registered. Please use a different email address.')

@user.route('/profile/<username>')
@login_required
def profile(username):
    user = User.query.filter_by(username=username).first_or_404()
    tweets = user.tweets.order_by(Tweet.timestamp.desc()).all()
    followers_count = user.followers.count()
    following_count = user.followed.count()
    is_following = current_user.is_following(user) if current_user.is_authenticated else False
    
    sentiment_stats = get_user_sentiment_stats(user)
    
    return render_template('profile/user.html',
                          title=f'{user.username}\'s Profile',
                          user=user,
                          tweets=tweets,
                          followers_count=followers_count,
                          following_count=following_count,
                          is_following=is_following,
                          sentiment_stats=sentiment_stats)

@user.route('/profile/edit', methods=['GET', 'POST'])
@login_required
def edit_profile():
    form = ProfileForm(current_user.username, current_user.email)
    if form.validate_on_submit():
        current_user.username = form.username.data
        current_user.email = form.email.data
        current_user.bio = form.bio.data
        db.session.commit()
        flash('Your profile has been updated.', 'success')
        return redirect(url_for('user.profile', username=current_user.username))
    elif request.method == 'GET':
        form.username.data = current_user.username
        form.email.data = current_user.email
        form.bio.data = current_user.bio
    
    return render_template('profile/edit.html', title='Edit Profile', form=form)

@user.route('/follow/<username>')
@login_required
def follow(username):
    user_to_follow = User.query.filter_by(username=username).first()
    
    if user_to_follow is None:
        flash(f'User {username} not found.', 'danger')
        return redirect(url_for('dashboard.index'))
    
    if user_to_follow == current_user:
        flash('You cannot follow yourself!', 'danger')
        return redirect(url_for('user.profile', username=username))
    
    current_user.follow(user_to_follow)
    db.session.commit()
    flash(f'You are now following {username}!', 'success')
    return redirect(url_for('user.profile', username=username))

@user.route('/unfollow/<username>')
@login_required
def unfollow(username):
    user_to_unfollow = User.query.filter_by(username=username).first()
    
    if user_to_unfollow is None:
        flash(f'User {username} not found.', 'danger')
        return redirect(url_for('dashboard.index'))
    
    if user_to_unfollow == current_user:
        flash('You cannot unfollow yourself!', 'danger')
        return redirect(url_for('user.profile', username=username))
    
    current_user.unfollow(user_to_unfollow)
    db.session.commit()
    flash(f'You have unfollowed {username}.', 'info')
    return redirect(url_for('user.profile', username=username))

@user.route('/followers/<username>')
@login_required
def followers(username):
    user = User.query.filter_by(username=username).first_or_404()
    followers = user.followers.all()
    follower_users = [follow.follower for follow in followers]
    
    return render_template('profile/followers.html', 
                          title=f'{user.username}\'s Followers',
                          user=user,
                          followers=follower_users)

@user.route('/following/<username>')
@login_required
def following(username):
    user = User.query.filter_by(username=username).first_or_404()
    following = user.followed.all()
    following_users = [follow.followee for follow in following]
    
    return render_template('profile/following.html',
                          title=f'People {user.username} Follows',
                          user=user,
                          following=following_users)